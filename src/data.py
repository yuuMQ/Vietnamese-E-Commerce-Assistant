from datasets import load_dataset, load_from_disk
from pprint import pprint

# DOWNLOAD Vietnamese Ecommerce dataset for Ecommerce assistant
# dataset = load_dataset("5CD-AI/Vietnamese-Ecommerce-Multi-turn-Chat")
# dataset.save_to_disk('../dataset')

dataset = load_from_disk('../dataset')
'''
DatasetDict({
    train: Dataset({
        features: ['id', 'conversations'],
        num_rows: 1482
    })
})
'''
pprint(dataset['train'][0])
'''
{'conversations': [{'from': 'human', 'value': 'Keo - Pad tản nhiệt là gì?'},
                   {'from': 'gpt',
                    'value': 'Keo - Pad tản nhiệt là một loại vật liệu được sử '
                             'dụng để lấp đầy khoảng hở giữa bộ xử lý và bộ '
                             'tản nhiệt, giúp cải thiện khả năng truyền nhiệt '
                             'từ bộ xử lý đến bộ tản nhiệt, từ đó giúp giảm '
                             'nhiệt độ của bộ xử lý.'},
                   {'from': 'human',
                    'value': 'Thành phần của keo - pad tản nhiệt là gì?'},
                   {'from': 'gpt',
                    'value': 'Keo - Pad tản nhiệt được làm từ nhiều loại vật '
                             'liệu khác nhau, bao gồm: chất làm ẩm, chất kết '
                             'dính, chất độn và chất làm tăng độ cứng. Thành '
                             'phần cụ thể của keo - pad tản nhiệt có thể thay '
                             'đổi tùy thuộc vào mục đích sử dụng.'},
                   {'from': 'human',
                    'value': 'Keo - pad tản nhiệt được sử dụng như thế nào?'},
                   {'from': 'gpt',
                    'value': 'Keo - Pad tản nhiệt được sử dụng bằng cách thoa '
                             'một lớp mỏng lên bề mặt của bộ xử lý, sau đó dán '
                             'bộ tản nhiệt lên trên. Lớp keo - pad tản nhiệt '
                             'sẽ lấp đầy khoảng hở giữa bộ xử lý và bộ tản '
                             'nhiệt, giúp cải thiện khả năng truyền nhiệt từ '
                             'bộ xử lý đến bộ tản nhiệt.'},
                   {'from': 'human',
                    'value': 'Những loại keo - pad tản nhiệt phổ biến là gì?'},
                   {'from': 'gpt',
                    'value': 'Có nhiều loại keo - pad tản nhiệt khác nhau trên '
                             'thị trường, bao gồm: keo - pad tản nhiệt '
                             'silicon, keo - pad tản nhiệt carbon, keo - pad '
                             'tản nhiệt kim loại lỏng và keo - pad tản nhiệt '
                             'silicon ceramic.'},
                   {'from': 'human',
                    'value': 'Keo - pad tản nhiệt silicon là gì?'},
                   {'from': 'gpt',
                    'value': 'Keo - Pad tản nhiệt silicon là một loại keo - '
                             'pad tản nhiệt được làm từ silicon, có độ bền '
                             'cao, khả năng dẫn nhiệt tốt và giá thành hợp '
                             'lý.'},
                   {'from': 'human',
                    'value': 'Keo - pad tản nhiệt carbon là gì?'},
                   {'from': 'gpt',
                    'value': 'Keo - Pad tản nhiệt carbon là một loại keo - pad '
                             'tản nhiệt được làm từ carbon, có khả năng dẫn '
                             'nhiệt tốt và độ bền cao, nhưng giá thành tương '
                             'đối cao.'},
                   {'from': 'human',
                    'value': 'Keo - pad tản nhiệt kim loại lỏng là gì?'},
                   {'from': 'gpt',
                    'value': 'Keo - Pad tản nhiệt kim loại lỏng là một loại '
                             'keo - pad tản nhiệt được làm từ kim loại lỏng, '
                             'có khả năng dẫn nhiệt tốt nhất trong các loại '
                             'keo - pad tản nhiệt, nhưng giá thành cao và có '
                             'thể gây ra nguy cơ rò rỉ nếu không sử dụng đúng '
                             'cách.'},
                   {'from': 'human',
                    'value': 'Keo - pad tản nhiệt silicon ceramic là gì?'},
                   {'from': 'gpt',
                    'value': 'Keo - Pad tản nhiệt silicon ceramic là một loại '
                             'keo - pad tản nhiệt được làm từ silicon và '
                             'ceramic, có khả năng dẫn nhiệt tốt, độ bền cao '
                             'và giá thành hợp lý.'},
                   {'from': 'human',
                    'value': 'Loại keo - pad tản nhiệt nào phù hợp với tôi?'},
                   {'from': 'gpt',
                    'value': 'Lựa chọn loại keo - pad tản nhiệt phù hợp phụ '
                             'thuộc vào nhiều yếu tố, bao gồm: loại bộ xử lý, '
                             'loại bộ tản nhiệt, mức nhiệt độ hoạt động mong '
                             'muốn và ngân sách của bạn. Bạn nên tham khảo ý '
                             'kiến của chuyên gia hoặc đọc các bài đánh giá để '
                             'lựa chọn loại keo - pad tản nhiệt phù hợp nhất.'},
                   {'from': 'human',
                    'value': 'Tôi nên mua keo - pad tản nhiệt ở đâu?'},
                   {'from': 'gpt',
                    'value': 'Bạn có thể mua keo - pad tản nhiệt tại các cửa '
                             'hàng bán linh kiện máy tính hoặc các trang '
                             'thương mại điện tử. Tuy nhiên, bạn nên chọn mua '
                             'sản phẩm từ những nhà cung cấp uy tín để đảm bảo '
                             'chất lượng và tránh mua phải hàng giả, hàng kém '
                             'chất lượng.'}],
 'id': '0'}

'''

