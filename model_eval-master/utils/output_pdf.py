import json
import datetime
from reportlab.lib.styles import ParagraphStyle as PS
from reportlab.platypus import PageBreak, Image, Table, TableStyle, Spacer
from reportlab.platypus.paragraph import Paragraph
from reportlab.platypus.doctemplate import PageTemplate, BaseDocTemplate
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.platypus.frames import Frame
from reportlab.lib.units import cm, inch
from reportlab.lib import colors
from PIL import Image as ImageRead
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
pdfmetrics.registerFont(TTFont('song', 'SimHei.ttf')) 

def is_Chinese(word):
    for ch in word:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False

centered_ch = PS(name='centered',
    fontName='song',
    fontSize=16,
    leading=16,
    alignment=1,
    spaceAfter=20)
centered2_ch = PS(name='centered',
    fontName='song',
    fontSize=10,
    leading=16,
    alignment=1,
    spaceAfter=20)
app_ps_ch = PS(name='app_ps',
    fontName='song',
    fontSize=6,
    leftIndent=10)
h1_ch = PS(
    name='Heading1',
    fontName='song',
    fontSize=12,
    leading=16)
h2_ch = PS(name='Heading2',
    fontName='song',
    fontSize=10,
    leading=14)
tStyle_ch = TableStyle([
    ('FONTNAME', (0, 0), (-1, -1), 'song'),
    ('FONTSIZE',(0,0),(-1,-1),6),
    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ('TEXTCOLOR',(0,0),(-1,-1),colors.black),
    ('INNERGRID', (0,0), (-1,-1), 0.1, colors.gray),
    ('BOX', (0,0), (-1,-1), 0.1, colors.gray),
                ])

centered = PS(name='centered',
    fontSize=16,
    leading=16,
    alignment=1,
    spaceAfter=20)
centered2 = PS(name='centered',
    fontSize=10,
    leading=16,
    alignment=1,
    spaceAfter=20)
app_ps = PS(name='app_ps',
    fontSize=6,
    leftIndent=10)
h1 = PS(
    name='Heading1',
    fontName='Times-Bold',
    fontSize=12,
    leading=16)
h2 = PS(name='Heading2',
    fontName='Times-Bold',
    fontSize=10,
    leading=14)
tStyle = TableStyle([
    ('FONTSIZE',(0,0),(-1,-1),6),
    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ('TEXTCOLOR',(0,0),(-1,-1),colors.black),
    ('INNERGRID', (0,0), (-1,-1), 0.1, colors.gray),
    ('BOX', (0,0), (-1,-1), 0.1, colors.gray),
                ])
cStyle = PS(
    name='c',
    fontName='Times-Bold',
    fontSize=10,
    leading=16)
chStyle = PS(
    name='ch',
    fontName='song',
    fontSize=8,
    leading=16)

def doHeading(text, sty, bookmark=None):
    from hashlib import sha1
    # create bookmarkname
    bn = sha1((text + sty.name).encode("utf8")).hexdigest()
    # modify paragraph text to include an anchor point with name bn
    if bookmark is None:
        h = Paragraph(text + '<a name="%s"/>' % bn, sty)
    else:
        #h = Paragraph(text + f'<a name="{bn}"> <b href=#"{bookmark}"/b></a>', sty)
        h = Paragraph(text=f"<a name='{bn}'/><link href=#{bookmark}> {text} </link>", style=sty)
    # store the bookmark name on the flowable so afterFlowable can see this
    h._bookmarkName = bn
    return h


class MyDocTemplate(BaseDocTemplate):
    def __init__(self, filename, experiment_name='Experiment', **kw):
        self.allowSplitting = 0
        super(MyDocTemplate, self).__init__(filename, **kw)
        template = PageTemplate('normal', [Frame(2.5*cm, 2.5*cm, 15*cm, 25*cm, id='F1')])
        self.addPageTemplates(template)

        self.table_results = []
        self.img_results = []
        self.app_results = []
        self.story = []
        self.bn_list = []
        self.nums = 0
        toc = TableOfContents()
        if is_Chinese(experiment_name):
            self.story.append(Paragraph(f'<b>{experiment_name}</b>', centered_ch))
            toc.levelStyles = [
                PS(fontName='song', fontSize=12, name='TOCHeading1', leftIndent=20, firstLineIndent=-20, spaceBefore=10, leading=16),
                PS(fontName='song', fontSize=8, name='TOCHeading2', leftIndent=40, firstLineIndent=-20, spaceBefore=5, leading=12),
            ]
            self.table_tag = doHeading('表格结果', h1_ch)
            self.img_tag = doHeading('图片结果', h1_ch)
            self.app_tag = doHeading('附录', h1_ch)
            self.ch_flag = True
        else:
            self.story.append(Paragraph(f'<b>{experiment_name}</b>', centered))
            toc.levelStyles = [
                PS(fontName='Times-Bold', fontSize=16, name='TOCHeading1', leftIndent=20, firstLineIndent=-20, spaceBefore=10, leading=16),
                PS(fontSize=12, name='TOCHeading2', leftIndent=40, firstLineIndent=-20, spaceBefore=5, leading=12),
            ]
            self.table_tag = doHeading('Table Result', h1)
            self.img_tag = doHeading('Image Result', h1)
            self.app_tag = doHeading('Appendix', h1)
            self.ch_flag = False
        self.TableResultHeading_bn = self.table_tag._bookmarkName
        self.table_stype = tStyle_ch
        self.tablecell = PS(name='tablecell',
                            fontName='song',
                            fontSize=6,
                            )
        
        time_now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
        self.story.append(Paragraph(f'<b>{time_now}</b>', centered2))
        self.story.append(toc)

    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph):     
            styleName = flowable.style.name     
            if styleName == 'Heading1':         # 第一级标题
                text = flowable.getPlainText()  # 取出段落的文本
                pageNum = self.page             # 取出当前页
                E = [0, text, pageNum]
                bn = getattr(flowable,'_bookmarkName',None)
                if bn is not None: E.append(bn)
                self.notify('TOCEntry', tuple(E))
                key = str(hash(flowable))       
                self.canv.bookmarkPage(key)    
                self.canv.addOutlineEntry(text, key, level=0, closed=0)
            elif styleName == 'Heading2':         # 第二级标题
                text = flowable.getPlainText()  # 取出段落的文本
                pageNum = self.page             # 取出当前页
                E = [1, text, pageNum]
                bn = getattr(flowable,'_bookmarkName',None)
                if bn is not None: E.append(bn)
                self.notify('TOCEntry', tuple(E))
                key = str(hash(flowable))       
                self.canv.bookmarkPage(key)    
                self.canv.addOutlineEntry(text, key, level=1, closed=0)
            try:
                text = flowable.getPlainText() 
            except:
                return                         
            for phrase in ['uniform','depraved','finger', 'Fraudulin']:    
                if text.find(phrase) > -1:     
                    self.notify('IndexEntry', (phrase, self.page))

    def addItem(self, param_dict={}, result_dict={}, img_paths=[], exp_name='exp1', comments=None):
        if isinstance(param_dict, dict):
            param_dict = [param_dict]
        if isinstance(result_dict, dict):
            result_dict = [result_dict]
        self.addImgItem(img_paths, exp_name, param_dict, result_dict, comments)
        self.addTableItem(param_dict, result_dict, exp_name)
    
    def addTableItem(self, param_dict_list, result_dict_list, exp_name):
        row = 0
        for param_dict, result_dict in zip(param_dict_list, result_dict_list):
            if len(self.table_results) == 0:
                table_keys = ['exp_name'] + list(param_dict.keys()) + list(result_dict.keys())
                self.table_results.append(table_keys)
            if row==0:
                item_dict = {'exp_name': exp_name}
            else:
                item_dict = {'exp_name': ''}
            item_dict.update(param_dict)
            item_dict.update(result_dict)
            item_result = []
            for i, key in enumerate(self.table_results[0]):
                value = item_dict[key]
                if isinstance(value, float):
                    value = round(value, 4)
                else:
                    value = str(value)
                if i == 0 and row == 0:
                    # tag_num = self.nums
                    value = Paragraph(text=f'<a href=#{self.bn_list[-1]}> {value} </a>', style=self.tablecell)
                item_result.append(value)
            self.table_results.append(item_result)
            row += 1
    
    def addImgItem(self, img_paths, exp_name, param_dict_list, result_dict_list, comments=None):
        if is_Chinese(exp_name):
            h = doHeading(f"{exp_name}", h2_ch, bookmark=self.TableResultHeading_bn)
        else:
            h = doHeading(f"{exp_name}", h2, bookmark=self.TableResultHeading_bn)
        self.img_results.append(h)
        self.bn_list.append(h._bookmarkName)
        self.nums += 1
        
        if comments is not None:
            if is_Chinese(comments):
                h = self.img_results.append(Paragraph(f'<b>{comments}</b>', chStyle))
            else:
                h = self.img_results.append(Paragraph(f'<b>{comments}</b>', cStyle))

        item_result = []
        for i, (param_dict, result_dict) in enumerate(zip(param_dict_list, result_dict_list)):
            if i == 0:
                table_keys = list(param_dict.keys()) + list(result_dict.keys())
                item_result.append(table_keys)
            tmp = []
            item_dict = {}
            item_dict.update(param_dict)
            item_dict.update(result_dict)
            for key in table_keys:
                value = item_dict[key]
                if isinstance(value, float):
                    value = round(value, 4)
                else:
                    value = str(value)
                tmp.append(value)
            item_result.append(tmp)

        table = Table(item_result)
        table.setStyle(self.table_stype)
        self.img_results.append(table)
        self.img_results.append(Spacer(1, 0.1 * inch))

        for img_path in img_paths:
            img = ImageRead.open(img_path)
            wh_ratio = img.height/img.width
            h = max(int(6*wh_ratio),1)
            if h>9:
                self.img_results.append(Image(img_path, int(6/wh_ratio)*inch, 9*inch))
            else:
                self.img_results.append(Image(img_path, 6*inch, h*inch))
            # self.img_results.append(Image(img_path))
        self.img_results.append(PageBreak())

    def addAppendixTable(self, app_name, info_dict):
        if is_Chinese(app_name):
            self.app_results.append(doHeading(f"{app_name}", h2_ch))
        else:
            self.app_results.append(doHeading(f"{app_name}", h2))
        table_keys = list(info_dict.keys())
        item_result = [table_keys]
        tmp = []
        for key in table_keys:
            value = info_dict[key]
            if isinstance(value, float):
                value = round(value, 4)
            else:
                value = str(value)
            tmp.append(value)
        item_result.append(tmp)

        table = Table(item_result)
        table.setStyle(self.table_stype)
        self.app_results.append(table)
        self.app_results.append(Spacer(1, 0.1 * inch))

    def addAppendixDict(self, app_name, info_dict):
        if is_Chinese(app_name):
            self.app_results.append(doHeading(f"{app_name}", h2_ch))
        else:
            self.app_results.append(doHeading(f"{app_name}", h2))
        info = json.dumps(info_dict, ensure_ascii=False, indent=4).replace('\n','<br />\n')
        if is_Chinese(info):
            self.app_results.append(Paragraph(info, app_ps_ch))
        else:
            self.app_results.append(Paragraph(info, app_ps))
        self.app_results.append(Spacer(1, 0.1 * inch))

    def buildPdf(self):
        self.story.append(PageBreak())
        self.story.append(self.table_tag)
        table = Table(self.table_results)
        table.setStyle(self.table_stype)
        self.story.append(table)
        self.story.append(PageBreak())
        self.story.append(self.img_tag)
        self.story.append(Spacer(1, 0.1 * inch))
        self.story += self.img_results
        if len(self.app_results)>0:
            self.story.append(self.app_tag)
            self.story.append(Spacer(1, 0.1 * inch))
            self.story += self.app_results
        self.multiBuild(self.story)

if __name__ == '__main__':
    doc = MyDocTemplate('demo.pdf', experiment_name='Experiment')
    for i in range(3):
        exp_name = 'exp'+str(i)
        img_paths = ['/home/wpxu/auto_modeling_frame/result/figures/model_0.jpg', 
                     '/home/wpxu/auto_modeling_frame/result/figures/model_1.jpg', 
                     '/home/wpxu/auto_modeling_frame/result/figures/model_2.jpg'] 
        # img_paths = ['/home/wpxu/auto_modeling_frame/figs_pdf/year.png', ]
        param_dict = {'参数一': 0.100001, '参数二':0.200001, '参数三':0.300001}
        result_dict = {'AlphaRtnNC': 0.1999999999, 'AlphaSharpeNC':0.26666666666, 'AlphaDrawdownNC':0.3333333, 'Long_TOV':0.444444444444}
        param_dict = {}
        comments = '中文comment'
        doc.addItem(param_dict, result_dict, img_paths, exp_name, comments)
    doc.buildPdf()