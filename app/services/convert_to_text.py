import json
class convert_to_text():
    def __init__(self, result):
        self.result = result
    def convert_pagewise(self):

        processed_data = {}
        for page in self.result['pages']:
            page_number = f'Page_{page["page"]}'
            concat_text = ' '.join(line["text"] for line in page["lines"])
            processed_data[page_number] = concat_text
        # convert processed_data to json
        processed_data = json.dumps(processed_data)

        return processed_data


    
        
        
        
