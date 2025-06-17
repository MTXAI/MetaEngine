# coding=gb2312
from engine.human.agent.rag import CSVLoader, OCRDocLoader, OCRLoader, OCRPDFLoader, OCRPPTLoader, JSONLoader, \
    ChineseRecursiveTextSplitter, ChineseTextSplitter, AliTextSplitter

if __name__ == '__main__':
    csv_loader = CSVLoader(
        file_path="../test_data/test.csv"
    )
    csv = csv_loader.load()
    print(csv)

    doc_loader = OCRDocLoader(
        file_path="../test_data/ocr_test.docx"
    )
    doc = doc_loader.load()
    print(doc)

    img_loader = OCRLoader(
        file_path="../test_data/ocr_test.jpg"
    )
    img = img_loader.load()
    print(img)

    pdf_loader = OCRPDFLoader(
        file_path="../test_data/ocr_test.pdf"
    )
    pdf = pdf_loader.load()
    print(pdf)

    ppt_loader = OCRPPTLoader(
        file_path="../test_data/ocr_test.pptx"
    )
    ppt = ppt_loader.load()
    print(ppt)

    json_loader = JSONLoader(
        file_path="../test_data/test.json",
        jq_schema=".messages[]"
    )
    json = json_loader.load()
    print(json)

    text_splitter = ChineseRecursiveTextSplitter(
        keep_separator=True, is_separator_regex=True, chunk_size=50, chunk_overlap=0
    )
    ls = [
        """�й�����ó�����Ʊ��棨75ҳ����ǰ 10 ���£�һ��ó�׽����� 19.5 ����Ԫ������ 25.1%�� ��������������ٸ߳� 2.9 ���ٷֵ㣬ռ�������ܶ�� 61.7%����ȥ��ͬ������ 1.6 ���ٷֵ㡣���У�һ��ó�׳��� 10.6 ����Ԫ������ 25.3%��ռ�����ܶ�� 60.9%������ 1.5 ���ٷֵ㣻����8.9����Ԫ������24.9%��ռ�����ܶ��62.7%�� ���� 1.8 ���ٷֵ㡣�ӹ�ó�׽����� 6.8 ����Ԫ������ 11.8%�� ռ�������ܶ�� 21.5%������ 2.0 ���ٷֵ㡣���У������� �� 10.4%��ռ�����ܶ�� 24.3%������ 2.6 ���ٷֵ㣻������ �� 14.2%��ռ�����ܶ�� 18.0%������ 1.2 ���ٷֵ㡣���⣬ �Ա�˰������ʽ������ 3.96 ����Ԫ������ 27.9%�����У��� �� 1.47 ����Ԫ������ 38.9%������ 2.49 ����Ԫ������ 22.2%��ǰ�����ȣ��й�����ó�׼������ֿ�������̬�ơ����� �������ܶ� 37834.3 ��Ԫ������ 11.6%�����з������ 17820.9 ��Ԫ������ 27.3%������ 20013.4 ��Ԫ������ 0.5%�������� ��ʵ���������������״�ת������������������ڽ��� 26.8 ���ٷֵ㣬��������ó������½� 62.9%�� 2192.5 ��Ԫ���� ��ó�׽ṹ�����Ż���֪ʶ�ܼ��ͷ�������� 16917.7 ��Ԫ�� ���� 13.3%��ռ����������ܶ�ı��شﵽ 44.7%������ 0.7 ���ٷֵ㡣 �����й�����ó�׷�չ����������չ�� ȫ������������������ø��շֻ��Ӿ磬������Ʒ�۸� ���ǡ���Դ��ȱ���������ż����ﾭ�������ߵ�������ȷ� �ս�֯���ӡ�ͬʱҲҪ�������ҹ����ó�����õ�����û�� �ı䣬��ó��ҵ���Ժͻ���������ǿ����ҵ̬��ģʽ�ӿ췢 չ������ת�Ͳ������١���ҵ����Ӧ��������ս����ŷ�ȼӿ��̨����ҵ��Ǩ�� �������ٲ�ҵ����Ӧ���������֣������˾������ҵ����Ӧ ����ȫ��˫��������һ���ع������򻯡����������������� ����������͹�ԡ����繩Ӧ���㣬����ҵ��ȱо�����������ޡ� �˼۸���ȫ���ҵ����Ӧ������ѹ���� ȫ��ͨ�ͳ�����λ���С���Դ�۸����ǼӴ���Ҫ������ ��ͨ��ѹ��������ȫ�򾭼ø��յĲ�ȷ���ԡ��������н��� 10 �·�����������Ʒ�г�չ����ָ������Դ�۸��� 2021 �� ������ 80%�������Խ��� 2022 ��С�����ǡ�IMF ָ����ȫ ��ͨ�����з��ռӾ磬ͨ��ǰ�����ھ޴�ȷ���ԡ�""",
    ]
    # text = """"""
    for inum, text in enumerate(ls):
        print(inum)
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            print(chunk)

    text_splitter = ChineseTextSplitter()
    for text in text_splitter.split_text(ls[0]):
        print(text)

    text_splitter = AliTextSplitter()
    for text in text_splitter.split_text(ls[0]):
        print(text)

