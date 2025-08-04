class AdvancedTextSplitter:
    def split(self, text, chunk_size=512):
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
