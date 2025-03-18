import re
import numpy as np
from collections import defaultdict
from string import punctuation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class RougeEvaluator:
    def __init__(self):
        self.punctuation = punctuation + '«»—'

    def _preprocess_text(self, text):
        text = text.lower()
        text = ''.join(char for char in text if char not in self.punctuation)
        return text.split()

    def _get_ngrams(self, words, n):
        return [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]

    def _count_overlap(self, reference_ngrams, candidate_ngrams):
        overlap = 0
        for ngram in candidate_ngrams:
            if ngram in reference_ngrams:
                overlap += 1
        return overlap

    def calculate_rouge_n(self, reference, candidate, n=1):
        ref_words = self._preprocess_text(reference)
        cand_words = self._preprocess_text(candidate)

        ref_ngrams = set(self._get_ngrams(ref_words, n))
        cand_ngrams = self._get_ngrams(cand_words, n)

        if not ref_ngrams or not cand_ngrams:
            return 0.0

        overlap = self._count_overlap(ref_ngrams, cand_ngrams)

        precision = overlap / len(cand_ngrams) if cand_ngrams else 0
        recall = overlap / len(ref_ngrams) if ref_ngrams else 0

        if precision + recall == 0:
            return 0.0
        f1 = 2 * (precision * recall) / (precision + recall)

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def calculate_rouge_l(self, reference, candidate):
        ref_words = self._preprocess_text(reference)
        cand_words = self._preprocess_text(candidate)

        m, n = len(ref_words), len(cand_words)
        lcs_table = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_words[i - 1] == cand_words[j - 1]:
                    lcs_table[i][j] = lcs_table[i - 1][j - 1] + 1
                else:
                    lcs_table[i][j] = max(lcs_table[i - 1][j], lcs_table[i][j - 1])

        lcs_length = lcs_table[m][n]

        precision = lcs_length / len(cand_words) if cand_words else 0
        recall = lcs_length / len(ref_words) if ref_words else 0

        if precision + recall == 0:
            return 0.0
        f1 = 2 * (precision * recall) / (precision + recall)

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }


class MongolianTextRankSummarizer:
    def __init__(self):
        self.stop_words = set([
           'би', 'бид', 'бидэн', 'чи', 'та', 'тэр', 'тэд', 'энэ', 'тэдэн',
            'нь', 'таны', 'тэдний', 'бусад', 'өөр', 'өөрийн', 'юм', 'байна', 
            'байдаг', 'байж', 'байгаа', 'байсан', 'бол', 'болов', 'боловч',
            'болох', 'болно', 'болсон', 'болж', 'буюу', 'ба', 'бөгөөд', 'мөн',
            'л', 'өө', 'энэ', 'тэр', 'тэгэх', 'ингэх', 'тэгээд', 'ингээд', 
            'гэхдээ', 'гэхэд', 'гэхээр', 'дараа', 'дараах', 'дараахан', 'дараагийн',
            'өөрөөр', 'өөрсдөө', 'өөрийгөө', 'өөрчлөгдөх', 'өөрчлөгдсөн',
            'дараа нь', 'дараачийн', 'дараахан', 'дараагийн', 'зарим', 'заримдаа',
            'зарим нь', 'заримынх', 'харин', 'хэдий', 'хэдийгээр', 'хэрэв',
            'хэрхэн', 'хэрхэн яаж', 'хэзээ', 'яг', 'яггүй', 'их', 'баг', 'цөөн',
            'нь', 'минь', 'чинь', 'таны', 'тэдний', 'үүнийг', 'үүгээр', 'үл', 
            'улмаар', 'улсын', 'улс', 'улмаар', 'учир', 'учраас', 'хот', 
            'хувьд', 'хэд', 'хэдэн', 'хэдий', 'хэдийгээр', 'хийх', 'хийсэн',
            'хойно', 'хойш', 'холбоотой', 'холбогдолтой', 'хугацаанд', 
            'хүн', 'хүмүүс', 'ямар', 'ямарваа', 'яагаад', 'ялангуяа',
            'явдал', 'явц', 'явагдах', 'явагдсан', 'ядах', 'ядаж',
            'юу', 'юуг', 'юунд', 'юутай', 'юугаар', 'юуг нь', 'юу ч',
            'за', 'зайлшгүй', 'дор', 'дорх', 'дээр', 'доор', 'дунд',
            'ойр', 'ойролцоо', 'өөрчлөлт', 'үнэхээр', 'үнэндээ', 'бараг',
            'тэр үед', 'тэгэхээр', 'ингэхээр', 'хэрхэн', 'яаж', 'яах',
            'яг одоо', 'ягштал', 'шиг', 'шүү', 'шилдэг', 'чанар',
            'үг', 'үгүүд', 'үгс',
        ])
        self.punctuation = punctuation + '«»—'
        self.evaluator = RougeEvaluator()
        self.damping = 0.85  # PageRank damping factor
        self.max_iterations = 100
        self.convergence_threshold = 0.0001
        self.text_length = 0
        
        self.use_position_weighting = True
        self.position_weight_beginning = 1.5  # Weight for sentences at the beginning
        self.position_weight_end = 1.2  # Weight for sentences at the end
        self.position_weight_middle = 0.8  # Weight for sentences in the middle
        self.position_threshold_beginning = 0.2  # First 20% of document
        self.position_threshold_end = 0.8  # Last 20% of document
        
        # Redundancy removal parameters
        self.redundancy_threshold = 0.6  # Similarity threshold for removing redundant sentences
        self.max_sentence_length = 25  # Words - can be tuned for Mongolian text


    def preprocess_text(self, text):
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        self.text_length = len(sentences)
        processed_sentences = []

        for sentence in sentences:
            cleaned = sentence.lower()
            cleaned = re.sub(r'\s+', ' ', cleaned)
            cleaned = re.sub(r'[^\w\s]', '', cleaned)
            words = [word for word in cleaned.split() if word not in self.stop_words]
            processed_sentences.append((sentence, words))

        return processed_sentences, len(sentences)

    def sentence_similarity(self, s1_words, s2_words):
        if not s1_words or not s2_words:
            return 0.0
            
        # Convert to sets for quicker intersection calculation
        s1_set = set(s1_words)
        s2_set = set(s2_words)
        
        # Calculate Jaccard similarity
        intersection = len(s1_set.intersection(s2_set))
        union = len(s1_set) + len(s2_set) - intersection
        
        return intersection / union if union > 0 else 0.0

    def build_similarity_matrix(self, processed_sentences):
        n = len(processed_sentences)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:  # No need to calculate similarity with itself
                    similarity_matrix[i][j] = self.sentence_similarity(
                        processed_sentences[i][1], 
                        processed_sentences[j][1]
                    )
                    
        # Normalize rows
        for i in range(n):
            row_sum = similarity_matrix[i].sum()
            if row_sum > 0:
                similarity_matrix[i] = similarity_matrix[i] / row_sum
                
        return similarity_matrix
    
    def calculate_position_weights(self, num_sentences):
        """Calculate weights based on sentence position in the document."""
        position_weights = np.ones(num_sentences)
        
        if not self.use_position_weighting:
            return position_weights
            
        beginning_threshold = int(num_sentences * self.position_threshold_beginning)
        end_threshold = int(num_sentences * self.position_threshold_end)
        
        for i in range(num_sentences):
            if i <= beginning_threshold:
                # Beginning sentences get higher weight
                position_weights[i] = self.position_weight_beginning
            elif i >= end_threshold:
                # End sentences get medium-high weight
                position_weights[i] = self.position_weight_end
            else:
                # Middle sentences get slightly lower weight
                position_weights[i] = self.position_weight_middle
                
        return position_weights

    def textrank(self, similarity_matrix):
        n = len(similarity_matrix)
        
        # Initialize scores with equal probabilities
        scores = np.ones(n) / n
        
        # Calculate position weights
        position_weights = self.calculate_position_weights(n)
        
        # Power iteration
        for _ in range(self.max_iterations):
            prev_scores = scores.copy()
            
            # Apply TextRank formula with position weights
            for i in range(n):
                rank_sum = 0
                for j in range(n):
                    if i != j and similarity_matrix[j, i] > 0:
                        rank_sum += similarity_matrix[j, i] * prev_scores[j]
                
                # Apply position weight to the standard TextRank formula
                scores[i] = ((1 - self.damping) + self.damping * rank_sum) * position_weights[i]
            
            # Normalize scores after applying weights
            scores_sum = np.sum(scores)
            if scores_sum > 0:
                scores = scores / scores_sum
            
            # Check for convergence
            if np.sum(np.abs(scores - prev_scores)) < self.convergence_threshold:
                break
                
        return scores

    def extract_phrases(self, sentence):
        return [p.strip() for p in sentence.split(',') if p.strip()]

    def identify_paragraph_positions(self, text):
        """Identify paragraph boundaries and important structural positions."""
        paragraphs = text.split('\n\n')
        paragraph_sentences = []
        sentence_positions = {}  # Maps sentence index to position info
        
        current_sentence_idx = 0
        for p_idx, paragraph in enumerate(paragraphs):
            if not paragraph.strip():
                continue
                
            # Split paragraph into sentences
            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            for s_idx, sentence in enumerate(sentences):
                # Mark sentence position within paragraph
                if s_idx == 0:
                    position_type = "paragraph_start"
                elif s_idx == len(sentences) - 1:
                    position_type = "paragraph_end"
                else:
                    position_type = "paragraph_middle"
                
                # Add document-level position info
                if p_idx == 0 and s_idx == 0:
                    position_type += ",document_start"
                elif p_idx == len(paragraphs) - 1 and s_idx == len(sentences) - 1:
                    position_type += ",document_end"
                
                sentence_positions[current_sentence_idx] = {
                    "position_type": position_type,
                    "paragraph_idx": p_idx,
                    "paragraph_position": s_idx / max(1, len(sentences) - 1)  # Normalized position
                }
                
                current_sentence_idx += 1
                
        return sentence_positions
                
    def remove_redundant_sentences(self, selected_indices, processed_sentences, similarity_matrix):
        """Remove redundant sentences from the selected indices."""
        if len(selected_indices) <= 1:
            return selected_indices
            
        # Sort by score (descending) to keep the most important sentences
        ranked_indices = sorted(selected_indices, key=lambda i: -self.sentence_scores[i])
        filtered_indices = [ranked_indices[0]]  # Start with the highest scoring sentence
        
        for idx in ranked_indices[1:]:
            # Check if the current sentence is too similar to any already selected sentence
            is_redundant = False
            for selected_idx in filtered_indices:
                if similarity_matrix[idx, selected_idx] > self.redundancy_threshold:
                    is_redundant = True
                    break
                    
            if not is_redundant:
                filtered_indices.append(idx)
                
        # Sort by original position to maintain document flow
        return sorted(filtered_indices)
        
    def trim_long_sentences(self, processed_sentences, indices):
        """Trim sentences that are too long."""
        result = []
        
        for idx in indices:
            original, words = processed_sentences[idx]
            if len(words) > self.max_sentence_length:
                # Find a good stopping point (period, comma, etc.)
                trimmed = self.trim_sentence(original)
                result.append((trimmed, words[:self.max_sentence_length]))
            else:
                result.append((original, words))
                
        return result
        
    def trim_sentence(self, sentence):
        """Find a good stopping point in a long sentence."""
        # Try to find the last period or semicolon in the first 3/4 of the sentence
        max_pos = int(len(sentence) * 0.75)
        last_period = sentence[:max_pos].rfind('.')
        last_semicolon = sentence[:max_pos].rfind(';')
        
        if last_period > 0:
            return sentence[:last_period+1]
        elif last_semicolon > 0:
            return sentence[:last_semicolon+1]
        else:
            # If no good stopping point, just truncate with ellipsis
            words = sentence.split()
            if len(words) > self.max_sentence_length:
                return ' '.join(words[:self.max_sentence_length]) + '...'
            return sentence
            
    def postprocess_summary(self, summary):
        """Apply post-processing to make the summary more concise."""
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', summary)
        processed = []
        
        for sentence in sentences:
            # Remove filler phrases
            for filler in ['нэмж дурдахад', 'мөн түүнчлэн', 'тухайлбал', 'өөрөөр хэлбэл', 'өөр үгээр хэлбэл', 'жишээ нь']:
                sentence = re.sub(fr'{filler},?\s*', '', sentence, flags=re.IGNORECASE)
            
            # Remove redundant adjectives (simplistic approach)
            words = sentence.split()
            if len(words) > 5:  # Only process longer sentences
                filtered_words = []
                prev_was_adj = False
                
                for i, word in enumerate(words):
                    # Skip adjacent adjectives (simplified detection)
                    if i > 0 and word.endswith(('тай', 'тэй', 'лиг', 'лэг')) and prev_was_adj:
                        continue
                        
                    filtered_words.append(word)
                    prev_was_adj = word.endswith(('тай', 'тэй', 'лиг', 'лэг'))
                
                sentence = ' '.join(filtered_words)
            
            # Trim extremely long sentences
            if len(sentence.split()) > self.max_sentence_length:
                sentence = self.trim_sentence(sentence)
                
            processed.append(sentence)
            
        # Join back and ensure proper spacing
        result = ' '.join(processed)
        result = re.sub(r'\s+', ' ', result)
        result = re.sub(r'\s([.,:;?!])', r'\1', result)
        
        return result.strip()

    def summarize(self, text, ratio=0.3):
        processed_sentences, total_sentences = self.preprocess_text(text)
        num_sentences = max(2, int(np.ceil(total_sentences * ratio)))
        
        # Build the similarity matrix
        similarity_matrix = self.build_similarity_matrix(processed_sentences)
        
        # Apply TextRank algorithm
        self.sentence_scores = self.textrank(similarity_matrix)
        
        # Get sentence indices sorted by score in descending order
        ranked_indices = np.argsort(self.sentence_scores)[::-1]
        
        # Select top sentences
        selected_indices = ranked_indices[:num_sentences]
        
        # Remove redundant sentences
        filtered_indices = self.remove_redundant_sentences(selected_indices, processed_sentences, similarity_matrix)
        
        # Sort selected indices by their original position to maintain document flow
        filtered_indices = sorted(filtered_indices)
        
        # Trim long sentences
        trimmed_sentences = self.trim_long_sentences(processed_sentences, filtered_indices)
        
        # Construct the summary
        raw_summary = ' '.join([s[0] for s in trimmed_sentences])
        
        # Apply post-processing for conciseness
        concise_summary = self.postprocess_summary(raw_summary)
        
        return concise_summary

    def evaluate_summary(self, generated_summary, reference_summary):
        def safe_rouge(rouge_func):
            result = rouge_func(reference_summary, generated_summary)
            return result if isinstance(result, dict) else {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

        rouge_1 = safe_rouge(lambda ref, gen: self.evaluator.calculate_rouge_n(ref, gen, n=1))
        rouge_2 = safe_rouge(lambda ref, gen: self.evaluator.calculate_rouge_n(ref, gen, n=2))
        rouge_l = safe_rouge(lambda ref, gen: self.evaluator.calculate_rouge_l(ref, gen))

        return {
            'rouge-1': rouge_1,
            'rouge-2': rouge_2,
            'rouge-l': rouge_l
        }


# if __name__ == "__main__":
#     summarizer = MongolianTextRankSummarizer()
#     # data_files = ["essays.csv", "news.csv", "social.csv", "paper.csv"]  
#     # labels = ["Essays", "News", "Social", "Paper"]  
#     # dataframes = [pd.read_csv(file) for file in data_files]
#     # display_rouge_scores_multiple11(dataframes, summarizer, labels)
#     text = """
#     Монгол улс сүүлийн арван жилд эрдэм шинжилгээ, технологийн салбарт идэвхтэй шинэчлэл хийх, инновацийн төлөвлөгөө хэрэгжүүлэхэд анхаарал хандуулж, олон улсын хамтын ажиллагааг өргөжүүлж байна. Улсын их сургуулиуд, судалгааны төвүүд болон хувийн хэвшил хамтран дижитал шилжилт, хиймэл оюун ухаан, мэдээллийн технологийн дэвшлийг хүчтэйгээр дэмжиж, шинэлэг төсөл, судалгааг амжилттай хэрэгжүүлж байна. Технологийн салбарт эдийн засаг, боловсрол, нийгмийн олон салбарт шинэ шийдлүүдийг нэвтрүүлэх бодлого, хөрөнгө оруулалт болон хамтын ажиллагааны платформууд тус улс орны өрсөлдөх чадварыг нэмэгдүүлж, дэлхийн зах зээлд нэр хүндтэй тогтвортой системүүдийг бүтээхэд томоохон үүрэг гүйцэтгэж байна.
#     Нэмж дурдахад, инновацийн экосистем нь зөвхөн технологийн төслүүдийг дэмжихэд төдийгүй, боловсролын шинэ арга барил, сургалтын программ, мэргэжлийн ур чадварыг хөгжүүлэхэд ихээхэн анхаарал хандуулж байна. Улс доторх судалгааны дэвшил болон олон улсын хамтын ажиллагаа нь эдийн засгийн өсөлт, нийгмийн бүтээлч хөгжилд эерэг нөлөө үзүүлж, ирээдүйд Монгол улсын өрсөлдөх чадварыг бататгах үндсэн түлхүүр болж өгч байна. Мөн төр, хувийн хэвшил, олон улсын сангуудад хамрагдаж, олон чиглэлд шинэ санаа, технологийн шийдлүүдийг нэвтрүүлэх арга хэмжээ авч байна.
#     Эдгээр хүчин чармайлтууд нь улсын эдийн засаг, боловсрол, нийгмийн олон талт хөгжилд дэмжлэг үзүүлж, ирээдүйд тогтвортой өсөлтийг хангах гол хүчин зүйлсийн нэг болж байна. Судалгааны дэвшил, инновац болон олон улсын хамтын ажиллагааны үр дүнд Монгол улс олон салбарт шинэ сорилтуудыг даван туулах, дэлхийн тавцан дээр нэр хүндтэй, бат бөх системүүдийг бүтээхэд амжилт гаргаж байна.
#     """
    
#     reference_summary = """
#     Монгол улс сүүлийн арван жилд эрдэм шинжилгээ, технологийн салбарт идэвхтэй шинэчлэл хийх, инновацийн төлөвлөгөө хэрэгжүүлэхэд анхаарал хандуулж, олон улсын хамтын ажиллагааг өргөжүүлж байна. Улсын их сургуулиуд, судалгааны төвүүд болон хувийн хэвшил хамтран дижитал шилжилт, хиймэл оюун ухаан, мэдээллийн технологийн дэвшлийг хүчтэйгээр дэмжиж, шинэлэг төсөл, судалгааг амжилттай хэрэгжүүлж байна. Инновацийн экосистем нь зөвхөн технологийн төслүүдийг дэмжихэд төдийгүй, боловсролын шинэ арга барил, сургалтын программ, мэргэжлийн ур чадварыг хөгжүүлэхэд ихээхэн анхаарал хандуулж байна. Улс доторх судалгааны дэвшил болон олон улсын хамтын ажиллагаа нь эдийн засгийн өсөлт, нийгмийн бүтээлч хөгжилд эерэг нөлөө үзүүлж, ирээдүйд Монгол улсын өрсөлдөх чадварыг бататгах үндсэн түлхүүр болж өгч байна. Судалгааны дэвшил, инновац болон олон улсын хамтын ажиллагааны үр дүнд Монгол улс олон салбарт шинэ сорилтуудыг даван туулах, дэлхийн тавцан дээр нэр хүндтэй, бат бөх системүүдийг бүтээхэд амжилт гаргаж байна.
#     """
#     summary = summarizer.summarize(text, ratio=0.3)
#     scores = summarizer.evaluate_summary(summary, reference_summary)
    
#     print("Generated Summary:")
#     print(summary)
    
#     print("\nROUGE Scores:")
#     for metric, values in scores.items():
#         print(f"{metric}:")
#         print(f"  Precision: {values['precision']:.4f}")
#         print(f"  Recall:    {values['recall']:.4f}")
#         print(f"  F1:        {values['f1']:.4f}")
        
def summarize_text(text, sentences):
    summarizer = MongolianTextRankSummarizer()
    summary = summarizer.summarize(text, ratio = sentences/10)
    return summary
    
