import collections
import operator
import statistics
class eval():
    def __init__(self, num_classes, pred_label_list, actual_label_list, gen_label_list):
#         super().__init__(pred_label_list, actual_label_list, gen_label_list)
        self.preds = pred_label_list
        self.labels = actual_label_list
        self.gender = gen_label_list
        self.num_classes = num_classes

        self.class_list = list(range(self.num_classes))
        self.class_list
        job_0 = []
        job_1 = []
        job_2 = []
        job_3 = []
        job_4 = []
        job_5 = []
        job_6 = []
        job_7 = []
     

        self.class_dict = {0:job_0, 1:job_1, 2:job_2, 3:job_3, 4:job_4, 5:job_5, 6:job_6, 7:job_7}
#         class_dict

        for i in self.class_list:
            for ind, label in enumerate(self.labels):
                if label == i:
        #             print(i,preds[ind])
                    self.class_dict[i].append((self.labels[ind], self.preds[ind], self.gender[ind]))
            
            
    # computing true positive rate difference
    def TPR_nums(self, key):
        res_dict = self.class_dict[key]

        F_Y = []
        M_Y = []
        F_Y_predY = []
        M_Y_predY = []
        for i in res_dict:
            if i[2] == 0:
                F_Y.append(i)
                if i[1] == key:
                     F_Y_predY.append(i)
            else:
                M_Y.append(i)
                if i[1] == key:
                    M_Y_predY.append(i)


        return F_Y, M_Y, F_Y_predY, M_Y_predY


    def TPR_diff(self, key):
        print(key)
        F_Y, M_Y, F_Y_predY, M_Y_predY = self.TPR_nums(key)
        try:
            TPR_d = ((len(F_Y_predY)/len(F_Y)) - (len(M_Y_predY)/len(M_Y)))
#             return TPR_d
        except:
            print(len(F_Y_predY), F_Y, M_Y_predY, M_Y)
        return TPR_d
