"""
Moteur RAG (Retrieval-Augmented Generation) pour la mémoire contextuelle.
Ce module gère l'indexation vectorielle avec FAISS, la recherche de similarité,
la gestion de la mémoire de travail et la compression de contexte.
"""

import os
import json
import pickle
import gc
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime
from collections import defaultdict
from contextlib import contextmanager
import numpy as np
import faiss
import time
import threading

from core.llm_connector import MistralEmbedConnector, LLMFactory
from utils.logger import setup_logger
from config import default_config


class CompressionManager:
    """
    Gestionnaire de compression isolé pour éviter les dépendances circulaires.
    """
    
    def __init__(self, rag_engine: 'RAGEngine'):
        self.rag = rag_engine
        self.logger = rag_engine.logger
        self.is_compressing = False
        self._compression_lock = threading.Lock()
    
    @contextmanager
    def compression_context(self):
        """Context manager pour éviter la récursion."""
        with self._compression_lock:
            if self.is_compressing:
                yield False
                return
            self.is_compressing = True
        try:
            yield True
        finally:
            with self._compression_lock:
                self.is_compressing = False
    
    def should_compress(self) -> bool:
        """Détermine si une compression est nécessaire."""
        if self.is_compressing:
            return False
        
        # Seuil de sécurité minimum
        if self.rag.index.ntotal < 50:
            return False
        
        threshold_vectors = int(self.rag.max_vectors * self.rag.compression_threshold)
        return self.rag.index.ntotal >= threshold_vectors
    
    def compress(self) -> Dict[str, Any]:
        """
        Effectue la compression sans récursion.
        Retourne les statistiques de compression.
        """
        stats = {
            'vectors_before': self.rag.index.ntotal,
            'vectors_after': 0,
            'summaries_created': 0,
            'entries_preserved': 0,
            'compression_ratio': 0.0
        }
        
        with self.compression_context() as can_compress:
            if not can_compress:
                self.logger.warning("Compression déjà en cours, abandon")
                return stats
            
            self.logger.info(f"Début compression: {self.rag.index.ntotal} vecteurs")
            
            # Grouper les entrées
            groups = self._group_entries_for_compression()
            
            # Créer les résumés et collecter les entrées à préserver
            preserved_entries = []
            new_summaries = []
            
            for (agent, milestone), entries in groups.items():
                if len(entries) < 10:  # Seuil minimum augmenté
                    # Préserver ces entrées
                    preserved_entries.extend(entries)
                    continue
                
                # Créer un résumé pour ce groupe
                summary = self._create_summary(agent, milestone, entries)
                if summary:
                    new_summaries.append(summary)
                    stats['summaries_created'] += 1
                else:
                    # En cas d'échec, préserver les entrées
                    preserved_entries.extend(entries)
            
            # Toujours préserver les fichiers projet
            project_files = [
                (i, meta) for i, meta in enumerate(self.rag.metadata)
                if self._is_project_file(meta)
            ]
            preserved_entries.extend(project_files)
            
            stats['entries_preserved'] = len(preserved_entries)
            
            # Reconstruire l'index avec les entrées préservées et les résumés
            self._rebuild_index(preserved_entries, new_summaries)
            
            stats['vectors_after'] = self.rag.index.ntotal
            stats['compression_ratio'] = 1 - (stats['vectors_after'] / stats['vectors_before'])
            
            self.logger.info(f"Compression terminée: {stats['vectors_before']} → {stats['vectors_after']} "
                           f"({stats['compression_ratio']:.1%} de réduction)")
            
            return stats
    
    def _group_entries_for_compression(self) -> Dict[Tuple[str, str], List[Tuple[int, Dict]]]:
        """Groupe les entrées par agent et jalon."""
        groups = defaultdict(list)
        
        for i, meta in enumerate(self.rag.metadata):

           # Règle 1: Ne jamais grouper les fichiers projet pour les compresser.
            if self._is_project_file(meta):
                continue
            
            # Règle 2: Ne jamais grouper un résumé déjà existant.
            if meta.get('type') == 'summary':
                continue

            # Règle 3 (implicite): Tout le reste est groupable et donc compressible.
            agent = meta.get('agent_name', 'unknown')
            milestone = meta.get('milestone', 'general')
            groups[(agent, milestone)].append((i, meta))
        
        return groups
    
    def _is_project_file(self, metadata: Dict[str, Any]) -> bool:
        """Détermine si une entrée est un fichier projet à préserver."""
        if metadata.get('preserve'):
            return True
        
        source = metadata.get('source', '')
        folder = metadata.get('folder', '')
        file_type = metadata.get('type', '')
        
        return (
            file_type == 'project_file' or
            source.startswith(('src/', 'docs/')) or
            folder in ['src', 'docs']
        )
    
    def _create_summary(self, agent: str, milestone: str, entries: List[Tuple[int, Dict]]) -> Optional[Dict]:
        """Crée un résumé pour un groupe d'entrées."""
        try:
            # Extraire les textes
            texts = []
            for _, meta in entries[:20]:  # Limiter pour le prompt
                text = meta.get('chunk_text', '')
                if text:
                    texts.append(text)
            
            if not texts:
                return None
            
            # Utiliser le LLM pour créer un résumé
            llm = LLMFactory.create(model='mistral-small-latest')
            
            prompt = f"""
            Crée un résumé concis des activités suivantes de l'agent {agent} pour le jalon {milestone}:
            
            {chr(10).join(texts[:10])}
            
            Résume en 2-3 phrases les points clés, décisions importantes et résultats.
            Sois concis et factuel.
            """
            
            summary_text = llm.generate(
                prompt=prompt,
                temperature=0.3,
                max_tokens=200
            )
            
            # Créer le résumé avec métadonnées
            summary = {
                'text': f"Résumé {agent} - {milestone}: {summary_text}",
                'metadata': {
                    'type': 'summary',
                    'agent_name': agent,
                    'milestone': milestone,
                    'entries_summarized': len(entries),
                    'created_at': datetime.now().isoformat()
                }
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Erreur création résumé {agent}/{milestone}: {str(e)}")
            return None
    
    def _rebuild_index(self, preserved_entries: List[Tuple[int, Dict]], new_summaries: List[Dict]):
        """Reconstruit l'index avec les entrées préservées et les nouveaux résumés."""
        # Créer un nouvel index
        new_index = faiss.IndexHNSWFlat(self.rag.embedding_dimension, self.rag.hnsw_m)
        new_index.hnsw.efConstruction = self.rag.hnsw_ef_construction
        new_index.hnsw.efSearch = self.rag.hnsw_ef_search
        
        new_metadata = []
        
        # 1. Ajouter les entrées préservées
        if preserved_entries:
            # Trier par index pour maintenir l'ordre
            preserved_entries.sort(key=lambda x: x[0])
            
            vectors = []
            for idx, meta in preserved_entries:
                try:
                    vector = self.rag.index.reconstruct(idx)
                    vectors.append(vector)
                    new_metadata.append(meta)
                except Exception as e:
                    self.logger.error(f"Erreur récupération vecteur {idx}: {str(e)}")
            
            if vectors:
                vectors_array = np.array(vectors).astype('float32')
                new_index.add(vectors_array)
        
        # 2. Ajouter les nouveaux résumés (sans passer par index_document!)
        if new_summaries:
            summary_texts = [s['text'] for s in new_summaries]
            try:
                embeddings = self.rag.embedding_model.embed_texts(summary_texts)
                new_index.add(embeddings.astype('float32'))
                
                # Ajouter les métadonnées des résumés
                start_idx = len(new_metadata)
                for i, summary in enumerate(new_summaries):
                    meta = summary['metadata'].copy()
                    meta['chunk_id'] = start_idx + i
                    meta['chunk_text'] = summary['text'][:500]
                    new_metadata.append(meta)
                    
            except Exception as e:
                self.logger.error(f"Erreur ajout résumés: {str(e)}")
        
        # 3. Remplacer l'index et les métadonnées
        self.rag.index = new_index
        self.rag.metadata = new_metadata
        
        # 4. Sauvegarder
        self.rag._save_index()
        self.rag._save_metadata()
        
        # Vérification finale
        project_files_count = len([m for m in new_metadata if self._is_project_file(m)])
        self.logger.info(f"Index reconstruit: {new_index.ntotal} vecteurs, "
                        f"{project_files_count} fichiers projet préservés")


class RAGEngine:
    """
    Moteur RAG pour indexer et rechercher dans les logs et documents.
    Version corrigée avec architecture propre.
    """

    def __init__(self, project_name: str, embedding_model: Optional[str] = None):
        self.project_name = project_name
        self.logger = setup_logger(f"RAGEngine.{project_name}")

        # Charger la configuration RAG
        rag_config = default_config['rag']
        self.embedding_model_name = rag_config['embedding_model']
        self.chunk_size = rag_config['chunk_size']
        self.chunk_overlap = rag_config['chunk_overlap']
        self.top_k = rag_config['top_k']
        self.similarity_threshold = rag_config['similarity_threshold']
        
        self.max_vectors = rag_config['max_vectors']
        self.compression_threshold = rag_config['compression_threshold']
        self.proactive_queries = rag_config['proactive_queries']
        self.max_context_tokens = rag_config['max_context_tokens']
        self.min_confidence_score = rag_config['min_confidence_score']
        self.score_weights = rag_config['score_weights']
        self.source_bonus = rag_config['source_bonus']
        self.auto_index_enabled = rag_config['auto_index_enabled']
        self.auto_index_extensions = rag_config['auto_index_extensions']
        self.auto_index_folders = rag_config['auto_index_folders']
        self.working_memory_enabled = rag_config['working_memory_enabled']

        # Paramètres HNSW
        self.hnsw_m = 16
        self.hnsw_ef_construction = 40
        self.hnsw_ef_search = 16

        # Chemins
        self.base_path = Path("projects") / project_name / "data" / "rag"
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.index_path = self.base_path / "faiss_hnsw.index"
        self.metadata_path = self.base_path / "metadata.pkl"
        
        # Chemin pour la mémoire de travail
        self.working_memory_path = self.base_path / "working_memory"
        self.working_memory_path.mkdir(exist_ok=True)

        self._init_embedding_model()
        self.metadata: List[Dict[str, Any]] = []
        self._init_index()
        
        # Initialiser la mémoire de travail si activée
        if self.working_memory_enabled:
            self._init_working_memory()
        
        # Initialiser le gestionnaire de compression
        self.compression_manager = CompressionManager(self)
        
        # File d'attente pour compression différée
        self._compression_pending = False
        self._indexing_active = False

        self.logger.info(f"RAG Engine Phase 4.5 (corrigé) initialisé pour {project_name}")

    def _init_embedding_model(self):
        try:
            self.embedding_model = MistralEmbedConnector()
            self.embedding_dimension = self.embedding_model.embedding_dimension
            self.logger.info("Modèle Mistral Embed initialisé")
        except Exception as e:
            self.logger.error(f"Erreur initialisation Mistral Embed: {str(e)}")
            raise

    def _init_index(self):
        if self.index_path.exists():
            try:
                self.index = faiss.read_index(str(self.index_path))
                self._load_metadata()
                if self.index.ntotal != len(self.metadata):
                    self.logger.warning("Incohérence détectée, réinitialisation...")
                    self._create_new_index()
                    self.metadata = []
                    self._save_index()
                    self._save_metadata()
                else:
                    self.logger.info(f"Index HNSW chargé: {self.index.ntotal} vecteurs")
                    self.index.hnsw.efSearch = self.hnsw_ef_search
            except Exception as e:
                self.logger.error(f"Erreur lors du chargement de l'index: {str(e)}")
                self._create_new_index()
        else:
            self._create_new_index()

    def _create_new_index(self):
        self.index = faiss.IndexHNSWFlat(self.embedding_dimension, self.hnsw_m)
        self.index.hnsw.efConstruction = self.hnsw_ef_construction
        self.index.hnsw.efSearch = self.hnsw_ef_search
        self.logger.info(f"Nouvel index HNSW créé (M={self.hnsw_m}, efConstruction={self.hnsw_ef_construction})")

    def _init_working_memory(self):
        """Initialise la mémoire de travail partagée entre tous les agents."""
        self.working_memory_index_path = self.working_memory_path / "wm_index.faiss"
        self.working_memory_metadata_path = self.working_memory_path / "wm_metadata.pkl"
        
        if self.working_memory_index_path.exists():
            try:
                self.working_memory_index = faiss.read_index(str(self.working_memory_index_path))
                with open(self.working_memory_metadata_path, 'rb') as f:
                    self.working_memory_metadata = pickle.load(f)
                self.logger.info(f"Mémoire de travail chargée: {self.working_memory_index.ntotal} vecteurs")
            except Exception as e: 
                self.logger.warning(f"Impossible de charger la mémoire de travail existante (raison: {e}).Une nouvelle mémoire sera créée.")
                self._create_new_working_memory()

        else:
            self._create_new_working_memory()

    def _create_new_working_memory(self):
        """Crée une nouvelle mémoire de travail."""
        self.working_memory_index = faiss.IndexHNSWFlat(self.embedding_dimension, self.hnsw_m)
        self.working_memory_metadata = []
        self.logger.info("Nouvelle mémoire de travail créée")

    def _load_metadata(self):
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
                self.logger.info(f"Métadonnées chargées: {len(self.metadata)} entrées")
            except Exception as e:
                self.logger.error(f"Erreur lors du chargement des métadonnées: {str(e)}")
                self.metadata = []

    def _save_metadata(self):
        try:
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)
        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde des métadonnées: {str(e)}")

    def _save_index(self):
        try:
            faiss.write_index(self.index, str(self.index_path))
        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde de l'index: {str(e)}")

    def _save_working_memory(self):
        """Sauvegarde la mémoire de travail sur disque."""
        if not self.working_memory_enabled:
            return
        try:
            faiss.write_index(self.working_memory_index, str(self.working_memory_index_path))
            with open(self.working_memory_metadata_path, 'wb') as f:
                pickle.dump(self.working_memory_metadata, f)
        except Exception as e:
            self.logger.error(f"Erreur sauvegarde mémoire de travail: {str(e)}")

    def _chunk_text(self, text: str) -> List[str]:
        """Découpe un texte en chunks avec overlap."""
        if not text or not text.strip():
            return []
            
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            
            # Essayer de couper à une phrase complète
            if end < text_length:
                last_period = text.rfind('.', start, end)
                last_newline = text.rfind('\n', start, end)
                cut_point = max(last_period, last_newline)
                
                if cut_point > start:
                    end = cut_point + 1
            
            chunk = text[start:end].strip()
            if len(chunk) > 0: 
                chunks.append(chunk)
            
            # Avancer avec overlap
            start = end - self.chunk_overlap if end < text_length else end
        
        return chunks

    def _index_raw(self, embeddings: np.ndarray, metadata_list: List[Dict[str, Any]], 
                   to_working_memory: bool = False) -> int:
        """
        Méthode privée pour indexer directement sans déclencher de compression.
        Utilisée par le CompressionManager pour éviter la récursion.
        """
        if to_working_memory and self.working_memory_enabled:
            start_idx = self.working_memory_index.ntotal
            self.working_memory_index.add(embeddings)
            
            for i, meta in enumerate(metadata_list):
                meta['chunk_id'] = start_idx + i
                meta['indexed_at'] = datetime.now().isoformat()
                meta['in_working_memory'] = True
                self.working_memory_metadata.append(meta)
            
            self._save_working_memory()
            return len(metadata_list)
        else:
            start_idx = self.index.ntotal
            self.index.add(embeddings)
            
            for i, meta in enumerate(metadata_list):
                meta['chunk_id'] = start_idx + i
                meta['indexed_at'] = datetime.now().isoformat()
                self.metadata.append(meta)
            
            return len(metadata_list)

    def index_document(self, content: str, metadata: Dict[str, Any], to_working_memory: bool = False) -> int:
        """
        Indexe un document avec gestion intelligente de la compression.
        """
        if not content or not content.strip():
            return 0
            
        chunks = self._chunk_text(content)
        if not chunks:
            return 0
        
        # Limiter le nombre de chunks
        max_chunks = 20
        if len(chunks) > max_chunks:
            self.logger.warning(f"Document trop grand ({len(chunks)} chunks), limité à {max_chunks}")
            chunks = chunks[:max_chunks]
        
        # Détection automatique de préservation
        source = metadata.get('source', '')
        folder = metadata.get('folder', '')
        
        is_project_file = (
            source.startswith('src/') or 
            source.startswith('docs/') or 
            folder in ['src', 'docs']
        )
        
        if is_project_file:
            metadata['preserve'] = True
        
        try:
            # Marquer l'indexation comme active
            self._indexing_active = True
            
            # Générer les embeddings
            embeddings = self.embedding_model.embed_texts(chunks)
            
            # Préparer les métadonnées
            metadata_list = []
            for i, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    'chunk_text': chunk,
                    'chunk_index': i,
                    'total_chunks': len(chunks)
                })
                metadata_list.append(chunk_metadata)
            
            # Indexer avec la méthode raw
            count = self._index_raw(embeddings, metadata_list, to_working_memory)
            
            # Sauvegarder périodiquement
            if not to_working_memory and len(self.metadata) % 100 == 0:
                self._save_index()
                self._save_metadata()
                gc.collect()
            
            self.logger.info(f"Document indexé: {count} chunks")
            
            # Vérifier si compression nécessaire (après indexation)
            if not to_working_memory and self.compression_manager.should_compress():
                self._compression_pending = True
            
            return count
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'indexation: {str(e)}")
            return 0
        finally:
            self._indexing_active = False
            
            # Effectuer la compression différée si nécessaire
            if self._compression_pending and not self._indexing_active:
                self._compression_pending = False
                self.trigger_compression_if_needed()

    def trigger_compression_if_needed(self):
        """Déclenche la compression si nécessaire, de manière sûre."""
        if self.compression_manager.should_compress():
            self.logger.info("Déclenchement de la compression différée")
            self.compression_manager.compress()

    def index_log_entry(self, log_entry: Dict[str, Any]) -> bool:
        """Indexe une entrée de log."""
        try:
            text_parts = []
            key_fields = ['message', 'interaction_type', 'content_summary']
            for field in key_fields:
                if field in log_entry and log_entry[field]:
                    text_parts.append(f"{field}: {str(log_entry[field])[:100]}")
            
            if not text_parts:
                return False
            
            content = " | ".join(text_parts)[:300]
            metadata = {
                'type': 'log_entry',
                'timestamp': log_entry.get('timestamp', datetime.now().isoformat())[:19],
                'agent_name': log_entry.get('agent_name', 'unknown')[:50],
                'milestone': log_entry.get('milestone', 'general')
            }
            
            # Les logs vont dans le RAG principal
            self.index_document(content, metadata, to_working_memory=False)
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'indexation du log: {str(e)}")
            return False

    def index_to_working_memory(self, content: str, metadata: Dict[str, Any]) -> int:
        """Indexe spécifiquement dans la mémoire de travail."""
        if not self.working_memory_enabled:
            self.logger.warning("Mémoire de travail désactivée")
            return 0
        
        metadata['memory_type'] = 'working'
        return self.index_document(content, metadata, to_working_memory=True)

    def index_project_files(self) -> int:
        """
        Indexe automatiquement les fichiers docs/ et src/ avec préservation.
        """
        if not self.auto_index_enabled:
            self.logger.info("Indexation automatique désactivée")
            return 0
        
        project_path = Path("projects") / self.project_name
        indexed_count = 0
        
        for folder in self.auto_index_folders:
            folder_path = project_path / folder
            if not folder_path.exists():
                continue
            
            for file_path in folder_path.rglob('*'):
                if file_path.is_file() and file_path.suffix in self.auto_index_extensions:
                    try:
                        content = file_path.read_text(encoding='utf-8')
                        if content.strip():
                            metadata = {
                                'type': 'project_file',
                                'source': str(file_path.relative_to(project_path)),
                                'file_type': file_path.suffix,
                                'folder': folder,
                                'preserve': True  # Marquer pour préservation automatique
                            }
                            chunks = self.index_document(content, metadata)
                            indexed_count += chunks
                            self.logger.info(f"Fichier indexé et préservé: {file_path.name} ({chunks} chunks)")
                    except Exception as e:
                        self.logger.error(f"Erreur indexation {file_path}: {str(e)}")
        
        self.logger.info(f"Indexation projet terminée: {indexed_count} chunks préservés au total")
        return indexed_count

    def search(self, query: str, top_k: Optional[int] = None, filter_metadata: Optional[Dict[str, Any]] = None,
              include_working_memory: bool = True) -> List[Dict[str, Any]]:
        """
        Recherche dans le RAG principal et optionnellement la mémoire de travail.
        """
        if not query:
            return []
            
        results = []
        k = min(top_k or self.top_k, max(self.index.ntotal, 1))
        
        # Recherche dans le RAG principal
        if self.index.ntotal > 0:
            try:
                query_embedding = self.embedding_model.embed_texts([query[:500]])[0:1]
                distances, indices = self.index.search(query_embedding, k)
                
                for distance, idx in zip(distances[0], indices[0]):
                    if idx >= 0 and idx < len(self.metadata):
                        similarity_score = max(0, 1 - (distance / 2))
                        if similarity_score >= self.similarity_threshold:
                            metadata = self.metadata[idx].copy()
                            metadata['score'] = float(similarity_score)
                            metadata['from_working_memory'] = False
                            
                            if filter_metadata:
                                match = all(
                                    metadata.get(key) == value
                                    for key, value in filter_metadata.items()
                                )
                                if not match:
                                    continue
                            
                            results.append(metadata)
            except Exception as e:
                self.logger.error(f"Erreur recherche RAG principal: {str(e)}")
        
        # Recherche dans la mémoire de travail
        if include_working_memory and self.working_memory_enabled and self.working_memory_index.ntotal > 0:
            try:
                query_embedding = self.embedding_model.embed_texts([query[:500]])[0:1]
                wm_k = min(top_k or self.top_k, self.working_memory_index.ntotal)
                distances, indices = self.working_memory_index.search(query_embedding, wm_k)
                
                for distance, idx in zip(distances[0], indices[0]):
                    if idx >= 0 and idx < len(self.working_memory_metadata):
                        similarity_score = max(0, 1 - (distance / 2))
                        if similarity_score >= self.similarity_threshold:
                            metadata = self.working_memory_metadata[idx].copy()
                            metadata['score'] = float(similarity_score)
                            metadata['from_working_memory'] = True
                            
                            if filter_metadata:
                                match = all(
                                    metadata.get(key) == value
                                    for key, value in filter_metadata.items()
                                )
                                if not match:
                                    continue
                            
                            results.append(metadata)
            except Exception as e:
                self.logger.error(f"Erreur recherche mémoire de travail: {str(e)}")
        
        # Appliquer le scoring avec pondération
        results = self._apply_confidence_scoring(results)
        
        # Trier par score final
        results.sort(key=lambda x: x.get('final_score', x.get('score', 0)), reverse=True)
        
        self.logger.info(f"Recherche '{query[:50]}...': {len(results)} résultats")
        return results[:top_k] if top_k else results

    def _apply_confidence_scoring(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Applique le scoring de confiance avec similarité, fraîcheur et source.
        """
        if not results:
            return results
        
        # Calculer l'âge relatif basé sur la position (plus simple que temporel)
        total_entries = len(self.metadata) + len(self.working_memory_metadata) if self.working_memory_enabled else len(self.metadata)
        
        for result in results:
            # Score de similarité (déjà calculé)
            similarity = result.get('score', 0)
            
            # Score de fraîcheur basé sur la position relative
            chunk_id = result.get('chunk_id', 0)
            if result.get('from_working_memory', False):
                # La mémoire de travail est toujours fraîche
                freshness = 1.0
            else:
                # Fraîcheur basée sur la position dans l'index
                freshness = chunk_id / total_entries if total_entries > 0 else 0.5
            
            # Score de source
            source_type = result.get('type', 'unknown')
            source_score = self.source_bonus.get(source_type, 0.5)
            
            # Calcul du score final pondéré
            final_score = (
                self.score_weights['similarity'] * similarity +
                self.score_weights['freshness'] * freshness +
                self.score_weights['source'] * source_score
            )
            
            result['final_score'] = final_score
            result['similarity_score'] = similarity
            result['freshness_score'] = freshness
            result['source_score'] = source_score
            
            # Filtrer par score minimum
            if final_score < self.min_confidence_score:
                result['below_threshold'] = True
        
        # Retirer les résultats sous le seuil
        return [r for r in results if not r.get('below_threshold', False)]

    def get_proactive_context(self, task_description: str, deliverables: List[str] = None, 
                            agent_name: str = None) -> str:
        """
        Génère un contexte enrichi avec recherche proactive pour un agent.
        """
        context_parts = []
        total_tokens = 0
        max_tokens = self.max_context_tokens
        
        # 1. Requête sur la tâche principale
        task_results = self.search(task_description, top_k=3)
        if task_results:
            context_parts.append("=== Contexte lié à la tâche ===")
            for r in task_results[:2]:
                text = r.get('chunk_text', '')[:150]
                if total_tokens + len(text.split()) < max_tokens:
                    context_parts.append(text)
                    total_tokens += len(text.split())
        
        # 2. Requête sur les livrables si fournis
        if deliverables and total_tokens < max_tokens:
            for deliverable in deliverables[:2]:
                del_results = self.search(deliverable, top_k=2)
                if del_results:
                    context_parts.append(f"\n=== Contexte pour {deliverable} ===")
                    for r in del_results[:1]:
                        text = r.get('chunk_text', '')[:100]
                        if total_tokens + len(text.split()) < max_tokens:
                            context_parts.append(text)
                            total_tokens += len(text.split())
        
        # 3. Contexte spécifique à l'agent
        if agent_name and total_tokens < max_tokens:
            agent_results = self.search(
                task_description,
                top_k=2,
                filter_metadata={'agent_name': agent_name}
            )
            if agent_results:
                context_parts.append(f"\n=== Historique {agent_name} ===")
                for r in agent_results:
                    text = r.get('chunk_text', '')[:100]
                    if total_tokens + len(text.split()) < max_tokens:
                        context_parts.append(text)
                        total_tokens += len(text.split())
        
        # 4. Mémoire de travail prioritaire
        if self.working_memory_enabled and total_tokens < max_tokens:
            wm_results = self.search(
                task_description, 
                top_k=2,
                include_working_memory=True
            )
            wm_only = [r for r in wm_results if r.get('from_working_memory', False)]
            if wm_only:
                context_parts.insert(0, "=== Mémoire de travail active ===")
                for i, r in enumerate(wm_only):
                    text = r.get('chunk_text', '')[:150]
                    if total_tokens + len(text.split()) < max_tokens:
                        context_parts.insert(i + 1, text)
                        total_tokens += len(text.split())
        
        return "\n".join(context_parts)

    def merge_working_memory_to_main(self, milestone_id: str = None):
        """
        Fusionne la mémoire de travail dans le RAG principal.
        """
        if not self.working_memory_enabled or self.working_memory_index.ntotal == 0:
            return
        
        self.logger.info(f"Fusion de la mémoire de travail ({self.working_memory_index.ntotal} vecteurs)")
        
        # Collecter les vecteurs et métadonnées
        vectors = []
        metadata_list = []
        
        for i in range(self.working_memory_index.ntotal):
            vector = self.working_memory_index.reconstruct(i)
            vectors.append(vector)
            
            meta = self.working_memory_metadata[i].copy()
            meta['from_working_memory'] = True
            if milestone_id:
                meta['milestone'] = milestone_id
            metadata_list.append(meta)
        
        if vectors:
            # Utiliser _index_raw pour éviter la récursion
            vectors_array = np.array(vectors).astype('float32')
            self._index_raw(vectors_array, metadata_list, to_working_memory=False)
        
        # Vider la mémoire de travail
        self._create_new_working_memory()
        self._save_working_memory()
        
        # Sauvegarder le RAG principal
        self._save_index()
        self._save_metadata()
        
        self.logger.info("Mémoire de travail fusionnée et vidée")

    def clear_working_memory(self):
        """Vide la mémoire de travail."""
        if self.working_memory_enabled:
            self._create_new_working_memory()
            self._save_working_memory()
            self.logger.info("Mémoire de travail vidée")

    def get_memory_usage(self) -> Dict[str, Any]:
        """Retourne l'utilisation mémoire du RAG."""
        wm_vectors = self.working_memory_index.ntotal if self.working_memory_enabled else 0
        
        return {
            'total_vectors': self.index.ntotal,
            'working_memory_vectors': wm_vectors,
            'metadata_count': len(self.metadata),
            'working_memory_metadata_count': len(self.working_memory_metadata) if self.working_memory_enabled else 0,
            'estimated_memory_mb': (self.index.ntotal * self.embedding_dimension * 4) / (1024 * 1024),
            'compression_threshold_reached': self.compression_manager.should_compress(),
            'usage_percentage': (self.index.ntotal / self.max_vectors * 100) if self.max_vectors > 0 else 0
        }

    def create_summary(self) -> Dict[str, Any]:
        """Génère un résumé statistique du contenu indexé."""
        if not self.metadata:
            return {
                'total_chunks': 0,
                'types': {},
                'agents': {},
                'working_memory_active': self.working_memory_enabled,
                'estimated_memory_mb': 0.0,
                'last_update': datetime.now().isoformat()
            }
        
        # Analyser les métadonnées
        types_count = defaultdict(int)
        agents_count = defaultdict(int)
        
        for meta in self.metadata:
            if 'type' in meta:
                types_count[meta['type']] += 1
            if 'agent_name' in meta:
                agents_count[meta['agent_name']] += 1
        
        # Statistiques mémoire de travail
        wm_stats = {}
        if self.working_memory_enabled and self.working_memory_metadata:
            wm_types = defaultdict(int)
            for meta in self.working_memory_metadata:
                if 'type' in meta:
                    wm_types[meta['type']] += 1
            
            wm_stats = {
                'chunks': len(self.working_memory_metadata),
                'types': dict(wm_types)
            }
        
        return {
            'total_chunks': len(self.metadata),
            'types': dict(types_count),
            'agents': dict(agents_count),
            'working_memory_active': self.working_memory_enabled,
            'working_memory_stats': wm_stats,
            'estimated_memory_mb': round((self.index.ntotal * self.embedding_dimension * 4) / (1024 * 1024), 2),
            'compression_needed': self.compression_manager.should_compress(),
            'last_update': datetime.now().isoformat()
        }

    def save_all(self):
        """Sauvegarde tous les index et métadonnées."""
        self._save_index()
        self._save_metadata()
        if self.working_memory_enabled:
            self._save_working_memory()
        self.logger.info("Tous les index et métadonnées sauvegardés")

    def clear_index(self):
        """Vide complètement le RAG (utile pour les tests)."""
        self._create_new_index()
        self.metadata = []
        if self.working_memory_enabled:
            self._create_new_working_memory()
        self.save_all()
        gc.collect()
        self.logger.info("RAG complètement vidé et mémoire libérée")