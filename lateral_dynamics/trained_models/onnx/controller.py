import torch

def nnodely_basic_model_update_state(data_in, rel):
    data_out = data_in.clone()
    max_dim = min(rel.size(1), data_in.size(1))
    data_out[:, -max_dim:, :] = rel[:, -max_dim:, :]
    return data_out

def nnodely_basic_model_timeshift(data_in):
    return torch.cat((data_in[:, 1:, :], data_in[:, :1, :]), dim=1)

def nnodely_layers_fuzzify_slicing(res, i, x):
    res[:, :, i:i+1] = x

def nnodely_layers_parametricfunction_understeer_corr_local_control(vx,curv,  # inputs
                                  A,        # constant
                                  ):
    return curv * (1 + A * torch.pow(vx, 2))

class TracerModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.all_parameters = {}
        self.all_constants = {}
        self._tensor_constant0 = torch.tensor(0.0)
        self._tensor_constant1 = torch.tensor(1.0)
        self._tensor_constant10 = torch.tensor(0.0)
        self._tensor_constant11 = torch.tensor(3)
        self._tensor_constant12 = torch.tensor(0.0)
        self._tensor_constant13 = torch.tensor(0.0)
        self._tensor_constant14 = torch.tensor(4)
        self._tensor_constant15 = torch.tensor(0.0)
        self._tensor_constant16 = torch.tensor(0.0)
        self._tensor_constant17 = torch.tensor(5)
        self._tensor_constant18 = torch.tensor(0.0)
        self._tensor_constant19 = torch.tensor(0.0)
        self._tensor_constant2 = torch.tensor(0)
        self._tensor_constant20 = torch.tensor(6)
        self._tensor_constant21 = torch.tensor(0.0)
        self._tensor_constant22 = torch.tensor(1.0)
        self._tensor_constant23 = torch.tensor(7)
        self._tensor_constant24 = torch.tensor(0.0)
        self._tensor_constant25 = torch.tensor(1.0)
        self._tensor_constant26 = torch.tensor(0)
        self._tensor_constant27 = torch.tensor(0.0)
        self._tensor_constant28 = torch.tensor(0.0)
        self._tensor_constant29 = torch.tensor(1)
        self._tensor_constant3 = torch.tensor(0.0)
        self._tensor_constant30 = torch.tensor(0.0)
        self._tensor_constant31 = torch.tensor(0.0)
        self._tensor_constant32 = torch.tensor(2)
        self._tensor_constant33 = torch.tensor(0.0)
        self._tensor_constant34 = torch.tensor(0.0)
        self._tensor_constant35 = torch.tensor(3)
        self._tensor_constant36 = torch.tensor(0.0)
        self._tensor_constant37 = torch.tensor(1.0)
        self._tensor_constant38 = torch.tensor(4)
        self._tensor_constant4 = torch.tensor(0.0)
        self._tensor_constant5 = torch.tensor(1)
        self._tensor_constant6 = torch.tensor(0.0)
        self._tensor_constant7 = torch.tensor(0.0)
        self._tensor_constant8 = torch.tensor(2)
        self._tensor_constant9 = torch.tensor(0.0)
        self.all_constants["A"] = torch.tensor([0.12241744995117188, 0.13215704262256622, 0.13487723469734192, 0.13838686048984528, 0.14319394528865814], requires_grad=False)
        self.all_parameters["Fir_InitCondition"] = torch.nn.Parameter(torch.tensor([[0.006525383796542883], [0.00875847227871418], [0.010448054410517216], [0.012908076867461205], [0.015002826228737831], [0.017531618475914], [0.019823284819722176], [0.021777980029582977], [0.024017440155148506], [0.026054205372929573], [0.027914606034755707], [0.029888296499848366], [0.032254818826913834], [0.0341765321791172], [0.03526938706636429], [0.036691855639219284], [0.03778807446360588], [0.03864985331892967], [0.03948149457573891], [0.03999602794647217], [0.0399942509829998], [0.040165819227695465], [0.04021630436182022], [0.040774110704660416], [-0.1947874128818512], [-0.3086092174053192], [-0.41924601793289185], [-0.5684259533882141], [-0.6486200094223022]]), requires_grad=True)
        self.all_parameters["Fir_target_0"] = torch.nn.Parameter(torch.tensor([[1.268223762512207], [1.2123229503631592], [1.1553658246994019], [1.0974462032318115], [1.0386626720428467], [0.9791200757026672], [0.918936014175415], [0.8582255244255066], [0.7971155047416687], [0.7357306480407715], [0.6742057800292969], [0.6126734018325806], [0.551267683506012], [0.4901258051395416], [0.4293866455554962], [0.3691798746585846], [0.3096372187137604], [0.2508878707885742], [0.19305293262004852], [0.13624544441699982], [0.08057688176631927], [0.02614888921380043], [-0.026943854987621307], [-0.07861576229333878], [-0.1287839114665985], [-0.17737722396850586], [-0.22432851791381836], [-0.26958194375038147], [-0.3130873441696167], [-0.3548048138618469]]), requires_grad=True)
        self.all_parameters["Fir_target_1"] = torch.nn.Parameter(torch.tensor([[0.960491955280304], [0.9125733375549316], [0.8639686107635498], [0.8147790431976318], [0.7651225924491882], [0.7151148915290833], [0.6648743152618408], [0.6145253777503967], [0.5641880035400391], [0.5139835476875305], [0.46403375267982483], [0.41445299983024597], [0.3653540313243866], [0.3168480694293976], [0.26903676986694336], [0.22201743721961975], [0.17587825655937195], [0.13070034980773926], [0.08655763417482376], [0.04351973161101341], [0.001651134341955185], [-0.03898698091506958], [-0.07833989709615707], [-0.11635544151067734], [-0.1529885083436966], [-0.18820056319236755], [-0.22196045517921448], [-0.25423887372016907], [-0.2850145697593689], [-0.31426703929901123]]), requires_grad=True)
        self.all_parameters["Fir_target_2"] = torch.nn.Parameter(torch.tensor([[0.7338525056838989], [0.6963642239570618], [0.6581295728683472], [0.6192700862884521], [0.5799053311347961], [0.5401735901832581], [0.5002042055130005], [0.46013733744621277], [0.42010679841041565], [0.38025379180908203], [0.3407123386859894], [0.3016184866428375], [0.26309964060783386], [0.22527901828289032], [0.1882622241973877], [0.15214475989341736], [0.11700902134180069], [0.08292242884635925], [0.049941062927246094], [0.018109621480107307], [-0.01253552921116352], [-0.04196719452738762], [-0.07016776502132416], [-0.09713040292263031], [-0.12285871058702469], [-0.14736995100975037], [-0.1706903874874115], [-0.19285544753074646], [-0.2139039933681488], [-0.23388241231441498]]), requires_grad=True)
        self.all_parameters["Fir_target_3"] = torch.nn.Parameter(torch.tensor([[0.5367298722267151], [0.5039888620376587], [0.47102683782577515], [0.43798840045928955], [0.40502795577049255], [0.37230929732322693], [0.339994341135025], [0.30825158953666687], [0.27723926305770874], [0.24710842967033386], [0.2179965376853943], [0.19002407789230347], [0.16330058872699738], [0.13791409134864807], [0.11393507570028305], [0.09141774475574493], [0.07039348781108856], [0.05087587982416153], [0.03285600617527962], [0.01631322130560875], [0.001206329558044672], [-0.012517699040472507], [-0.024925772100687027], [-0.0360947884619236], [-0.04610498249530792], [-0.05503883585333824], [-0.0629836916923523], [-0.07002319395542145], [-0.07623922824859619], [-0.0817127451300621]]), requires_grad=True)
        self.all_parameters["Fir_target_4"] = torch.nn.Parameter(torch.tensor([[0.4531867206096649], [0.42839130759239197], [0.40343332290649414], [0.37835294008255005], [0.3531997501850128], [0.3280305564403534], [0.302904337644577], [0.27788183093070984], [0.2530285716056824], [0.2284020185470581], [0.20406456291675568], [0.1800747513771057], [0.15648838877677917], [0.1333637684583664], [0.1107538565993309], [0.08870929479598999], [0.06728191673755646], [0.04652145132422447], [0.0264738742262125], [0.007184571120887995], [-0.011301771737635136], [-0.02894427813589573], [-0.04570416733622551], [-0.061547573655843735], [-0.07644730061292648], [-0.09038344770669937], [-0.10334645956754684], [-0.11533810943365097], [-0.12637034058570862], [-0.13646841049194336]]), requires_grad=True)
        self.all_parameters["Fir_target_5"] = torch.nn.Parameter(torch.tensor([[0.3500901162624359], [0.33111560344696045], [0.3120191991329193], [0.29282182455062866], [0.2735459506511688], [0.25420239567756653], [0.23463675379753113], [0.2137681394815445], [0.19446726143360138], [0.1753213256597519], [0.15635381639003754], [0.1376185417175293], [0.11917741596698761], [0.10110028833150864], [0.08346159756183624], [0.0663311704993248], [0.04978550598025322], [0.03389427065849304], [0.018721355125308037], [0.004327945876866579], [-0.009233424440026283], [-0.021925609558820724], [-0.0337493009865284], [-0.045058026909828186], [-0.05592691898345947], [-0.06472314894199371], [-0.07262793183326721], [-0.0796075239777565], [-0.08566644042730331], [-0.09082780033349991]]), requires_grad=True)
        self.all_parameters["Fir_target_6"] = torch.nn.Parameter(torch.tensor([[0.3691418766975403], [0.3361518085002899], [0.3038986325263977], [0.27263134717941284], [0.2425866574048996], [0.21387435495853424], [0.1842704862356186], [0.15837490558624268], [0.13467274606227875], [0.11300656199455261], [0.09333977848291397], [0.0756470113992691], [0.05988377332687378], [0.04598277434706688], [0.03385661542415619], [0.023399431258440018], [0.014495929703116417], [0.0070196157321333885], [0.0008421638049185276], [-0.004159446805715561], [-0.00809982605278492], [-0.011088019236922264], [-0.013226036913692951], [-0.01460599061101675], [-0.015312421135604382], [-0.015418552793562412], [-0.014988959766924381], [-0.01407924760133028], [-0.012737021781504154], [-0.011004920117557049]]), requires_grad=True)
        self.all_parameters["Fir_target_7"] = torch.nn.Parameter(torch.tensor([[0.26424068212509155], [0.23828989267349243], [0.2138243466615677], [0.19077646732330322], [0.16910751163959503], [0.14878997206687927], [0.1298121064901352], [0.11217080056667328], [0.09586340934038162], [0.08089044690132141], [0.0672445297241211], [0.05490420013666153], [0.043833643198013306], [0.033983293920755386], [0.0252943467348814], [0.017701735720038414], [0.01113949529826641], [0.0055382344871759415], [0.0008285960648208857], [-0.003057369962334633], [-0.0061882296577095985], [-0.008632984943687916], [-0.010457922704517841], [-0.011723444797098637], [-0.012489711865782738], [-0.01281608734279871], [-0.012759726494550705], [-0.012372310273349285], [-0.011700070463120937], [-0.010769709013402462]]), requires_grad=True)
        self.all_constants["SamplePart100"] = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], requires_grad=True)
        self.all_constants["SamplePart106"] = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], requires_grad=True)
        self.all_constants["SamplePart109"] = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], requires_grad=True)
        self.all_constants["SamplePart111"] = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], requires_grad=True)
        self.all_constants["SamplePart97"] = torch.tensor([[1.0]], requires_grad=True)
        self.all_constants["Select114"] = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select117"] = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select120"] = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select123"] = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select126"] = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select129"] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], requires_grad=True)
        self.all_constants["Select132"] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], requires_grad=True)
        self.all_constants["Select135"] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], requires_grad=True)
        self.all_parameters = torch.nn.ParameterDict(self.all_parameters)
        self.all_constants = torch.nn.ParameterDict(self.all_constants)

    def update(self, closed_loop={}, connect={}, disconnect=False):
        pass
    def forward(self, kwargs):
        getitem = kwargs['controller_vx_in']
        relation_forward_sample_part100_w = self.all_constants.SamplePart100
        einsum = torch.functional.einsum('bij,ki->bkj', getitem, relation_forward_sample_part100_w);  getitem = relation_forward_sample_part100_w = None
        zeros_like = torch.zeros_like(einsum)
        repeat = zeros_like.repeat(1, 1, 8);  zeros_like = None
        sub = einsum - 8.0
        neg = -sub;  sub = None
        truediv = neg / 2.1428571428571423;  neg = None
        add = truediv + 1;  truediv = None
        _tensor_constant0 = self._tensor_constant0
        maximum = torch.maximum(add, _tensor_constant0);  add = _tensor_constant0 = None
        _tensor_constant1 = self._tensor_constant1
        minimum = torch.minimum(maximum, _tensor_constant1);  maximum = _tensor_constant1 = None
        _tensor_constant2 = self._tensor_constant2
        slicing = nnodely_layers_fuzzify_slicing(repeat, _tensor_constant2, minimum);  _tensor_constant2 = minimum = None
        sub_1 = einsum - 8.0
        truediv_1 = sub_1 / 2.1428571428571423;  sub_1 = None
        _tensor_constant3 = self._tensor_constant3
        maximum_1 = torch.maximum(truediv_1, _tensor_constant3);  truediv_1 = _tensor_constant3 = None
        sub_2 = einsum - 10.142857142857142
        neg_1 = -sub_2;  sub_2 = None
        truediv_2 = neg_1 / 2.1428571428571423;  neg_1 = None
        add_1 = truediv_2 + 1;  truediv_2 = None
        _tensor_constant4 = self._tensor_constant4
        maximum_2 = torch.maximum(add_1, _tensor_constant4);  add_1 = _tensor_constant4 = None
        minimum_1 = torch.minimum(maximum_1, maximum_2);  maximum_1 = maximum_2 = None
        _tensor_constant5 = self._tensor_constant5
        slicing_1 = nnodely_layers_fuzzify_slicing(repeat, _tensor_constant5, minimum_1);  _tensor_constant5 = minimum_1 = None
        sub_3 = einsum - 10.142857142857142
        truediv_3 = sub_3 / 2.1428571428571423;  sub_3 = None
        _tensor_constant6 = self._tensor_constant6
        maximum_3 = torch.maximum(truediv_3, _tensor_constant6);  truediv_3 = _tensor_constant6 = None
        sub_4 = einsum - 12.285714285714285
        neg_2 = -sub_4;  sub_4 = None
        truediv_4 = neg_2 / 2.142857142857144;  neg_2 = None
        add_2 = truediv_4 + 1;  truediv_4 = None
        _tensor_constant7 = self._tensor_constant7
        maximum_4 = torch.maximum(add_2, _tensor_constant7);  add_2 = _tensor_constant7 = None
        minimum_2 = torch.minimum(maximum_3, maximum_4);  maximum_3 = maximum_4 = None
        _tensor_constant8 = self._tensor_constant8
        slicing_2 = nnodely_layers_fuzzify_slicing(repeat, _tensor_constant8, minimum_2);  _tensor_constant8 = minimum_2 = None
        sub_5 = einsum - 12.285714285714285
        truediv_5 = sub_5 / 2.142857142857144;  sub_5 = None
        _tensor_constant9 = self._tensor_constant9
        maximum_5 = torch.maximum(truediv_5, _tensor_constant9);  truediv_5 = _tensor_constant9 = None
        sub_6 = einsum - 14.428571428571429
        neg_3 = -sub_6;  sub_6 = None
        truediv_6 = neg_3 / 2.1428571428571406;  neg_3 = None
        add_3 = truediv_6 + 1;  truediv_6 = None
        _tensor_constant10 = self._tensor_constant10
        maximum_6 = torch.maximum(add_3, _tensor_constant10);  add_3 = _tensor_constant10 = None
        minimum_3 = torch.minimum(maximum_5, maximum_6);  maximum_5 = maximum_6 = None
        _tensor_constant11 = self._tensor_constant11
        slicing_3 = nnodely_layers_fuzzify_slicing(repeat, _tensor_constant11, minimum_3);  _tensor_constant11 = minimum_3 = None
        sub_7 = einsum - 14.428571428571429
        truediv_7 = sub_7 / 2.1428571428571406;  sub_7 = None
        _tensor_constant12 = self._tensor_constant12
        maximum_7 = torch.maximum(truediv_7, _tensor_constant12);  truediv_7 = _tensor_constant12 = None
        sub_8 = einsum - 16.57142857142857
        neg_4 = -sub_8;  sub_8 = None
        truediv_8 = neg_4 / 2.142857142857146;  neg_4 = None
        add_4 = truediv_8 + 1;  truediv_8 = None
        _tensor_constant13 = self._tensor_constant13
        maximum_8 = torch.maximum(add_4, _tensor_constant13);  add_4 = _tensor_constant13 = None
        minimum_4 = torch.minimum(maximum_7, maximum_8);  maximum_7 = maximum_8 = None
        _tensor_constant14 = self._tensor_constant14
        slicing_4 = nnodely_layers_fuzzify_slicing(repeat, _tensor_constant14, minimum_4);  _tensor_constant14 = minimum_4 = None
        sub_9 = einsum - 16.57142857142857
        truediv_9 = sub_9 / 2.142857142857146;  sub_9 = None
        _tensor_constant15 = self._tensor_constant15
        maximum_9 = torch.maximum(truediv_9, _tensor_constant15);  truediv_9 = _tensor_constant15 = None
        sub_10 = einsum - 18.714285714285715
        neg_5 = -sub_10;  sub_10 = None
        truediv_10 = neg_5 / 2.1428571428571423;  neg_5 = None
        add_5 = truediv_10 + 1;  truediv_10 = None
        _tensor_constant16 = self._tensor_constant16
        maximum_10 = torch.maximum(add_5, _tensor_constant16);  add_5 = _tensor_constant16 = None
        minimum_5 = torch.minimum(maximum_9, maximum_10);  maximum_9 = maximum_10 = None
        _tensor_constant17 = self._tensor_constant17
        slicing_5 = nnodely_layers_fuzzify_slicing(repeat, _tensor_constant17, minimum_5);  _tensor_constant17 = minimum_5 = None
        sub_11 = einsum - 18.714285714285715
        truediv_11 = sub_11 / 2.1428571428571423;  sub_11 = None
        _tensor_constant18 = self._tensor_constant18
        maximum_11 = torch.maximum(truediv_11, _tensor_constant18);  truediv_11 = _tensor_constant18 = None
        sub_12 = einsum - 20.857142857142858
        neg_6 = -sub_12;  sub_12 = None
        truediv_12 = neg_6 / 2.1428571428571423;  neg_6 = None
        add_6 = truediv_12 + 1;  truediv_12 = None
        _tensor_constant19 = self._tensor_constant19
        maximum_12 = torch.maximum(add_6, _tensor_constant19);  add_6 = _tensor_constant19 = None
        minimum_6 = torch.minimum(maximum_11, maximum_12);  maximum_11 = maximum_12 = None
        _tensor_constant20 = self._tensor_constant20
        slicing_6 = nnodely_layers_fuzzify_slicing(repeat, _tensor_constant20, minimum_6);  _tensor_constant20 = minimum_6 = None
        sub_13 = einsum - 20.857142857142858;  einsum = None
        truediv_13 = sub_13 / 2.1428571428571423;  sub_13 = None
        _tensor_constant21 = self._tensor_constant21
        maximum_13 = torch.maximum(truediv_13, _tensor_constant21);  truediv_13 = _tensor_constant21 = None
        _tensor_constant22 = self._tensor_constant22
        minimum_7 = torch.minimum(maximum_13, _tensor_constant22);  maximum_13 = _tensor_constant22 = None
        _tensor_constant23 = self._tensor_constant23
        slicing_7 = nnodely_layers_fuzzify_slicing(repeat, _tensor_constant23, minimum_7);  _tensor_constant23 = minimum_7 = None
        relation_forward_select135_w = self.all_constants.Select135
        einsum_1 = torch.functional.einsum('ijk,k->ij', repeat, relation_forward_select135_w);  relation_forward_select135_w = None
        unsqueeze = einsum_1.unsqueeze(2);  einsum_1 = None
        getitem_1 = kwargs['controller_ax_in']
        relation_forward_sample_part97_w = self.all_constants.SamplePart97
        einsum_2 = torch.functional.einsum('bij,ki->bkj', getitem_1, relation_forward_sample_part97_w);  getitem_1 = relation_forward_sample_part97_w = None
        zeros_like_1 = torch.zeros_like(einsum_2)
        repeat_1 = zeros_like_1.repeat(1, 1, 5);  zeros_like_1 = None
        sub_14 = einsum_2 - -2.0
        neg_7 = -sub_14;  sub_14 = None
        truediv_14 = neg_7 / 1.0;  neg_7 = None
        add_7 = truediv_14 + 1;  truediv_14 = None
        _tensor_constant24 = self._tensor_constant24
        maximum_14 = torch.maximum(add_7, _tensor_constant24);  add_7 = _tensor_constant24 = None
        _tensor_constant25 = self._tensor_constant25
        minimum_8 = torch.minimum(maximum_14, _tensor_constant25);  maximum_14 = _tensor_constant25 = None
        _tensor_constant26 = self._tensor_constant26
        slicing_8 = nnodely_layers_fuzzify_slicing(repeat_1, _tensor_constant26, minimum_8);  _tensor_constant26 = minimum_8 = None
        sub_15 = einsum_2 - -2.0
        truediv_15 = sub_15 / 1.0;  sub_15 = None
        _tensor_constant27 = self._tensor_constant27
        maximum_15 = torch.maximum(truediv_15, _tensor_constant27);  truediv_15 = _tensor_constant27 = None
        sub_16 = einsum_2 - -1.0
        neg_8 = -sub_16;  sub_16 = None
        truediv_16 = neg_8 / 1.0;  neg_8 = None
        add_8 = truediv_16 + 1;  truediv_16 = None
        _tensor_constant28 = self._tensor_constant28
        maximum_16 = torch.maximum(add_8, _tensor_constant28);  add_8 = _tensor_constant28 = None
        minimum_9 = torch.minimum(maximum_15, maximum_16);  maximum_15 = maximum_16 = None
        _tensor_constant29 = self._tensor_constant29
        slicing_9 = nnodely_layers_fuzzify_slicing(repeat_1, _tensor_constant29, minimum_9);  _tensor_constant29 = minimum_9 = None
        sub_17 = einsum_2 - -1.0
        truediv_17 = sub_17 / 1.0;  sub_17 = None
        _tensor_constant30 = self._tensor_constant30
        maximum_17 = torch.maximum(truediv_17, _tensor_constant30);  truediv_17 = _tensor_constant30 = None
        sub_18 = einsum_2 - 0.0
        neg_9 = -sub_18;  sub_18 = None
        truediv_18 = neg_9 / 1.0;  neg_9 = None
        add_9 = truediv_18 + 1;  truediv_18 = None
        _tensor_constant31 = self._tensor_constant31
        maximum_18 = torch.maximum(add_9, _tensor_constant31);  add_9 = _tensor_constant31 = None
        minimum_10 = torch.minimum(maximum_17, maximum_18);  maximum_17 = maximum_18 = None
        _tensor_constant32 = self._tensor_constant32
        slicing_10 = nnodely_layers_fuzzify_slicing(repeat_1, _tensor_constant32, minimum_10);  _tensor_constant32 = minimum_10 = None
        sub_19 = einsum_2 - 0.0
        truediv_19 = sub_19 / 1.0;  sub_19 = None
        _tensor_constant33 = self._tensor_constant33
        maximum_19 = torch.maximum(truediv_19, _tensor_constant33);  truediv_19 = _tensor_constant33 = None
        sub_20 = einsum_2 - 1.0
        neg_10 = -sub_20;  sub_20 = None
        truediv_20 = neg_10 / 1.0;  neg_10 = None
        add_10 = truediv_20 + 1;  truediv_20 = None
        _tensor_constant34 = self._tensor_constant34
        maximum_20 = torch.maximum(add_10, _tensor_constant34);  add_10 = _tensor_constant34 = None
        minimum_11 = torch.minimum(maximum_19, maximum_20);  maximum_19 = maximum_20 = None
        _tensor_constant35 = self._tensor_constant35
        slicing_11 = nnodely_layers_fuzzify_slicing(repeat_1, _tensor_constant35, minimum_11);  _tensor_constant35 = minimum_11 = None
        sub_21 = einsum_2 - 1.0;  einsum_2 = None
        truediv_21 = sub_21 / 1.0;  sub_21 = None
        _tensor_constant36 = self._tensor_constant36
        maximum_21 = torch.maximum(truediv_21, _tensor_constant36);  truediv_21 = _tensor_constant36 = None
        _tensor_constant37 = self._tensor_constant37
        minimum_12 = torch.minimum(maximum_21, _tensor_constant37);  maximum_21 = _tensor_constant37 = None
        _tensor_constant38 = self._tensor_constant38
        slicing_12 = nnodely_layers_fuzzify_slicing(repeat_1, _tensor_constant38, minimum_12);  _tensor_constant38 = minimum_12 = None
        all_constants_a = self.all_constants.A
        mul = repeat_1 * all_constants_a;  repeat_1 = all_constants_a = None
        sum_1 = torch.sum(mul, dim = 2, keepdim = True);  mul = None
        getitem_2 = kwargs['controller_curv_in']
        relation_forward_sample_part111_w = self.all_constants.SamplePart111
        einsum_3 = torch.functional.einsum('bij,ki->bkj', getitem_2, relation_forward_sample_part111_w);  getitem_2 = relation_forward_sample_part111_w = None
        getitem_3 = kwargs['controller_vx_in']
        relation_forward_sample_part109_w = self.all_constants.SamplePart109
        einsum_4 = torch.functional.einsum('bij,ki->bkj', getitem_3, relation_forward_sample_part109_w);  getitem_3 = relation_forward_sample_part109_w = None
        understeer_corr_local_control = nnodely_layers_parametricfunction_understeer_corr_local_control(einsum_4, einsum_3, sum_1);  einsum_4 = einsum_3 = sum_1 = None
        size = understeer_corr_local_control.size(0)
        relation_forward_fir134_weights = self.all_parameters.Fir_target_7
        size_1 = relation_forward_fir134_weights.size(1)
        squeeze = understeer_corr_local_control.squeeze(-1)
        matmul = torch.matmul(squeeze, relation_forward_fir134_weights);  squeeze = relation_forward_fir134_weights = None
        to = matmul.to(dtype = torch.float32);  matmul = None
        view = to.view(size, 1, size_1);  to = size = size_1 = None
        mul_1 = view * unsqueeze;  view = unsqueeze = None
        relation_forward_select132_w = self.all_constants.Select132
        einsum_5 = torch.functional.einsum('ijk,k->ij', repeat, relation_forward_select132_w);  relation_forward_select132_w = None
        unsqueeze_1 = einsum_5.unsqueeze(2);  einsum_5 = None
        size_2 = understeer_corr_local_control.size(0)
        relation_forward_fir131_weights = self.all_parameters.Fir_target_6
        size_3 = relation_forward_fir131_weights.size(1)
        squeeze_1 = understeer_corr_local_control.squeeze(-1)
        matmul_1 = torch.matmul(squeeze_1, relation_forward_fir131_weights);  squeeze_1 = relation_forward_fir131_weights = None
        to_1 = matmul_1.to(dtype = torch.float32);  matmul_1 = None
        view_1 = to_1.view(size_2, 1, size_3);  to_1 = size_2 = size_3 = None
        mul_2 = view_1 * unsqueeze_1;  view_1 = unsqueeze_1 = None
        relation_forward_select129_w = self.all_constants.Select129
        einsum_6 = torch.functional.einsum('ijk,k->ij', repeat, relation_forward_select129_w);  relation_forward_select129_w = None
        unsqueeze_2 = einsum_6.unsqueeze(2);  einsum_6 = None
        size_4 = understeer_corr_local_control.size(0)
        relation_forward_fir128_weights = self.all_parameters.Fir_target_5
        size_5 = relation_forward_fir128_weights.size(1)
        squeeze_2 = understeer_corr_local_control.squeeze(-1)
        matmul_2 = torch.matmul(squeeze_2, relation_forward_fir128_weights);  squeeze_2 = relation_forward_fir128_weights = None
        to_2 = matmul_2.to(dtype = torch.float32);  matmul_2 = None
        view_2 = to_2.view(size_4, 1, size_5);  to_2 = size_4 = size_5 = None
        mul_3 = view_2 * unsqueeze_2;  view_2 = unsqueeze_2 = None
        relation_forward_select126_w = self.all_constants.Select126
        einsum_7 = torch.functional.einsum('ijk,k->ij', repeat, relation_forward_select126_w);  relation_forward_select126_w = None
        unsqueeze_3 = einsum_7.unsqueeze(2);  einsum_7 = None
        size_6 = understeer_corr_local_control.size(0)
        relation_forward_fir125_weights = self.all_parameters.Fir_target_4
        size_7 = relation_forward_fir125_weights.size(1)
        squeeze_3 = understeer_corr_local_control.squeeze(-1)
        matmul_3 = torch.matmul(squeeze_3, relation_forward_fir125_weights);  squeeze_3 = relation_forward_fir125_weights = None
        to_3 = matmul_3.to(dtype = torch.float32);  matmul_3 = None
        view_3 = to_3.view(size_6, 1, size_7);  to_3 = size_6 = size_7 = None
        mul_4 = view_3 * unsqueeze_3;  view_3 = unsqueeze_3 = None
        relation_forward_select123_w = self.all_constants.Select123
        einsum_8 = torch.functional.einsum('ijk,k->ij', repeat, relation_forward_select123_w);  relation_forward_select123_w = None
        unsqueeze_4 = einsum_8.unsqueeze(2);  einsum_8 = None
        size_8 = understeer_corr_local_control.size(0)
        relation_forward_fir122_weights = self.all_parameters.Fir_target_3
        size_9 = relation_forward_fir122_weights.size(1)
        squeeze_4 = understeer_corr_local_control.squeeze(-1)
        matmul_4 = torch.matmul(squeeze_4, relation_forward_fir122_weights);  squeeze_4 = relation_forward_fir122_weights = None
        to_4 = matmul_4.to(dtype = torch.float32);  matmul_4 = None
        view_4 = to_4.view(size_8, 1, size_9);  to_4 = size_8 = size_9 = None
        mul_5 = view_4 * unsqueeze_4;  view_4 = unsqueeze_4 = None
        relation_forward_select120_w = self.all_constants.Select120
        einsum_9 = torch.functional.einsum('ijk,k->ij', repeat, relation_forward_select120_w);  relation_forward_select120_w = None
        unsqueeze_5 = einsum_9.unsqueeze(2);  einsum_9 = None
        size_10 = understeer_corr_local_control.size(0)
        relation_forward_fir119_weights = self.all_parameters.Fir_target_2
        size_11 = relation_forward_fir119_weights.size(1)
        squeeze_5 = understeer_corr_local_control.squeeze(-1)
        matmul_5 = torch.matmul(squeeze_5, relation_forward_fir119_weights);  squeeze_5 = relation_forward_fir119_weights = None
        to_5 = matmul_5.to(dtype = torch.float32);  matmul_5 = None
        view_5 = to_5.view(size_10, 1, size_11);  to_5 = size_10 = size_11 = None
        mul_6 = view_5 * unsqueeze_5;  view_5 = unsqueeze_5 = None
        relation_forward_select117_w = self.all_constants.Select117
        einsum_10 = torch.functional.einsum('ijk,k->ij', repeat, relation_forward_select117_w);  relation_forward_select117_w = None
        unsqueeze_6 = einsum_10.unsqueeze(2);  einsum_10 = None
        size_12 = understeer_corr_local_control.size(0)
        relation_forward_fir116_weights = self.all_parameters.Fir_target_1
        size_13 = relation_forward_fir116_weights.size(1)
        squeeze_6 = understeer_corr_local_control.squeeze(-1)
        matmul_6 = torch.matmul(squeeze_6, relation_forward_fir116_weights);  squeeze_6 = relation_forward_fir116_weights = None
        to_6 = matmul_6.to(dtype = torch.float32);  matmul_6 = None
        view_6 = to_6.view(size_12, 1, size_13);  to_6 = size_12 = size_13 = None
        mul_7 = view_6 * unsqueeze_6;  view_6 = unsqueeze_6 = None
        relation_forward_select114_w = self.all_constants.Select114
        einsum_11 = torch.functional.einsum('ijk,k->ij', repeat, relation_forward_select114_w);  repeat = relation_forward_select114_w = None
        unsqueeze_7 = einsum_11.unsqueeze(2);  einsum_11 = None
        size_14 = understeer_corr_local_control.size(0)
        relation_forward_fir113_weights = self.all_parameters.Fir_target_0
        size_15 = relation_forward_fir113_weights.size(1)
        squeeze_7 = understeer_corr_local_control.squeeze(-1);  understeer_corr_local_control = None
        matmul_7 = torch.matmul(squeeze_7, relation_forward_fir113_weights);  squeeze_7 = relation_forward_fir113_weights = None
        to_7 = matmul_7.to(dtype = torch.float32);  matmul_7 = None
        view_7 = to_7.view(size_14, 1, size_15);  to_7 = size_14 = size_15 = None
        mul_8 = view_7 * unsqueeze_7;  view_7 = unsqueeze_7 = None
        add_11 = mul_8 + mul_7;  mul_8 = mul_7 = None
        add_12 = add_11 + mul_6;  add_11 = mul_6 = None
        add_13 = add_12 + mul_5;  add_12 = mul_5 = None
        add_14 = add_13 + mul_4;  add_13 = mul_4 = None
        add_15 = add_14 + mul_3;  add_14 = mul_3 = None
        add_16 = add_15 + mul_2;  add_15 = mul_2 = None
        add_17 = add_16 + mul_1;  add_16 = mul_1 = None
        getitem_4 = kwargs['controller_steer_in'];  kwargs = None
        relation_forward_sample_part106_w = self.all_constants.SamplePart106
        einsum_12 = torch.functional.einsum('bij,ki->bkj', getitem_4, relation_forward_sample_part106_w);  getitem_4 = relation_forward_sample_part106_w = None
        size_16 = einsum_12.size(0)
        relation_forward_fir107_weights = self.all_parameters.Fir_InitCondition
        size_17 = relation_forward_fir107_weights.size(1)
        squeeze_8 = einsum_12.squeeze(-1);  einsum_12 = None
        matmul_8 = torch.matmul(squeeze_8, relation_forward_fir107_weights);  squeeze_8 = relation_forward_fir107_weights = None
        to_8 = matmul_8.to(dtype = torch.float32);  matmul_8 = None
        view_8 = to_8.view(size_16, 1, size_17);  to_8 = size_16 = size_17 = None
        add_18 = add_17 + view_8
        return ({'controller_steer_out': add_18, 'controller_steer_from_ic': view_8, 'controller_steer_from_target': add_17}, {}, {'controller_steer_in': add_18}, {})
        
class RecurrentModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Cell = TracerModel()
        self.inputs = ['controller_vx_in', 'controller_ax_in', 'controller_curv_in', ]
        self.states = dict()

    def forward(self, kwargs):
        n_samples = min([kwargs[key].size(0) for key in self.inputs])
        self.states['controller_steer_in'] = kwargs['controller_steer_in']
        results = {'controller_steer_out':[], 'controller_steer_from_ic':[], 'controller_steer_from_target':[], }
        X = dict()
        for idx in range(n_samples):
            for key in self.inputs:
                X[key] = kwargs[key][idx]
            for key, value in self.states.items():
                X[key] = value
            out, _, closed_loop, connect = self.Cell(X)
            for key, value in results.items():
                results[key].append(out[key])
            for key, val in closed_loop.items():
                self.states[key] = nnodely_basic_model_timeshift(self.states[key])
                self.states[key] = nnodely_basic_model_update_state(self.states[key], val)
            for key, val in connect.items():
                self.states[key] = nnodely_basic_model_timeshift(val)
        return results
