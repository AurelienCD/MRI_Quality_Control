# coding: utf-8

import os
import unittest
from slicer.ScriptedLoadableModule import *
import logging
from __main__ import vtk, qt, ctk, slicer
from math import *
import numpy as np
from vtk.util import numpy_support
import SimpleITK as sitk
import sitkUtils as su
import time
import codecs
import datetime
import vtkSegmentationCorePython as vtkSegmentationCore

date = datetime.datetime.now()

#
# MRIBasicQualityControl
#

class MRI_Quality_Control(ScriptedLoadableModule):

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "MRI Quality Control"
    self.parent.categories = ["Imaging"]
    self.parent.dependencies = []
    self.parent.contributors = ["Aurelien CORROYER-DULMONT PhD and Cyril JAUDET PhD"]
    self.parent.helpText = u"Le mode opératoire de ce CQ est disponible dans : //s-grp/grp/RADIOPHY/Personnel/Aurélien Corroyer-Dulmont/"+str(3)+u"dSlicer/Mode Opératoire" ### A FAIRE ###
    self.parent.acknowledgementText = "Medical Physics department, Centre Francois Baclesse, CAEN, FRANCE."


    # Additional initialization step after application startup is complete
    slicer.app.connect("startupCompleted()", registerSampleData)

#
# Register sample data sets in Sample Data module
#

def registerSampleData():
  """
  Add data sets to Sample Data module.
  """
  # It is always recommended to provide sample data for users to make it easy to try the module,
  # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

  import SampleData
  iconsPath = os.path.join(os.path.dirname(__file__), 'Resources/Icons')

  # To ensure that the source code repository remains small (can be downloaded and installed quickly)
  # it is recommended to store data sets that are larger than a few MB in a Github release.

  # MRI_Quality_Control1
  SampleData.SampleDataLogic.registerCustomSampleDataSource(
    # Category and sample name displayed in Sample Data module
    category='MRI_Quality_Control',
    sampleName='MRI_Quality_Control1',
    # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
    # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
    thumbnailFileName=os.path.join(iconsPath, 'MRI_Quality_Control1.png'),
    # Download URL and target file name
    uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
    fileNames='MRI_Quality_Control1.nrrd',
    # Checksum to ensure file integrity. Can be computed by this command:
    #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
    checksums = 'SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95',
    # This node name will be used when the data set is loaded
    nodeNames='MRI_Quality_Control1'
  )

  # MRI_Quality_Control2
  SampleData.SampleDataLogic.registerCustomSampleDataSource(
    # Category and sample name displayed in Sample Data module
    category='MRI_Quality_Control',
    sampleName='MRI_Quality_Control2',
    thumbnailFileName=os.path.join(iconsPath, 'MRI_Quality_Control2.png'),
    # Download URL and target file name
    uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
    fileNames='MRI_Quality_Control2.nrrd',
    checksums = 'SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97',
    # This node name will be used when the data set is loaded
    nodeNames='MRI_Quality_Control2'
  )

#
# MRI_Quality_ControlWidget
#

class MRI_Quality_ControlWidget(ScriptedLoadableModuleWidget):

  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)

    # Instantiate and connect widgets ...

    #
    # Parameters Area
    #
    parametersCollapsibleButton = ctk.ctkCollapsibleButton()
    parametersCollapsibleButton.text = "Parameters"
    self.layout.addWidget(parametersCollapsibleButton)

    # Layout within the dummy collapsible button
    parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)

    #
    # input volume selector - CT Image
    #
    self.inputSelectorCTImage = slicer.qMRMLNodeComboBox()
    self.inputSelectorCTImage.nodeTypes = ["vtkMRMLScalarVolumeNode"]
    self.inputSelectorCTImage.selectNodeUponCreation = True
    self.inputSelectorCTImage.addEnabled = False
    self.inputSelectorCTImage.removeEnabled = False
    self.inputSelectorCTImage.noneEnabled = False
    self.inputSelectorCTImage.showHidden = False
    self.inputSelectorCTImage.showChildNodeTypes = False
    self.inputSelectorCTImage.setMRMLScene( slicer.mrmlScene )
    self.inputSelectorCTImage.setToolTip( "Pick the input to the algorithm." )
    parametersFormLayout.addRow("Reference CT Image: ", self.inputSelectorCTImage)

    #
    # input volume selector - Spheres label
    #
    self.inputSelectorMaskImg = slicer.qMRMLNodeComboBox()
    self.inputSelectorMaskImg.nodeTypes = ["vtkMRMLLabelMapVolumeNode"]
    self.inputSelectorMaskImg.selectNodeUponCreation = True
    self.inputSelectorMaskImg.addEnabled = False
    self.inputSelectorMaskImg.removeEnabled = False
    self.inputSelectorMaskImg.noneEnabled = False
    self.inputSelectorMaskImg.showHidden = False
    self.inputSelectorMaskImg.showChildNodeTypes = False
    self.inputSelectorMaskImg.setMRMLScene( slicer.mrmlScene )
    self.inputSelectorMaskImg.setToolTip( "Pick the input to the algorithm." )
    parametersFormLayout.addRow("Spheres label (CT space): ", self.inputSelectorMaskImg)

    #
    # input volume selector - MRI Image
    #
    self.inputSelectorMRImage = slicer.qMRMLNodeComboBox()
    self.inputSelectorMRImage.nodeTypes = ["vtkMRMLScalarVolumeNode"]
    self.inputSelectorMRImage.selectNodeUponCreation = True
    self.inputSelectorMRImage.addEnabled = False
    self.inputSelectorMRImage.removeEnabled = False
    self.inputSelectorMRImage.noneEnabled = False
    self.inputSelectorMRImage.showHidden = False
    self.inputSelectorMRImage.showChildNodeTypes = False
    self.inputSelectorMRImage.setMRMLScene( slicer.mrmlScene )
    self.inputSelectorMRImage.setToolTip( "Pick the input to the algorithm." )
    parametersFormLayout.addRow("MR Image to analyse: ", self.inputSelectorMRImage)

    #
    # Output director Button
    #
    label = qt.QLabel('Directory output:')
    self.OutputDirectory = ctk.ctkDirectoryButton()
    self.OutputDirectory.directory = u'//s-grp/grp/Imagerie/CQ/IRM/Resultats_mensuel'
    self.OutputDirectory.caption = 'Output directory'
    parametersFormLayout.addRow(label, self.OutputDirectory)

    #
    # Registration Button
    #
    self.RegistrationButton = qt.QPushButton("1) Registration")
    self.RegistrationButton.toolTip = "Run the algorithm."
    self.RegistrationButton.enabled = False
    parametersFormLayout.addRow(self.RegistrationButton)

    #
    # Geometric distorsion Button
    #
    self.GeoDistorsionButton = qt.QPushButton("2A) Geometric distorsion analysis")
    self.GeoDistorsionButton.toolTip = "Run the geometric distorsion algorithm."
    self.GeoDistorsionButton.enabled = False
    parametersFormLayout.addRow(self.GeoDistorsionButton)

    #
    # input volume selector - geometric distorsion result volume
    #
    self.inputSelectorGeoDistorsionResult = slicer.qMRMLNodeComboBox()
    self.inputSelectorGeoDistorsionResult.nodeTypes = ["vtkMRMLScalarVolumeNode"]
    self.inputSelectorGeoDistorsionResult.selectNodeUponCreation = True
    self.inputSelectorGeoDistorsionResult.addEnabled = False
    self.inputSelectorGeoDistorsionResult.removeEnabled = False
    self.inputSelectorGeoDistorsionResult.noneEnabled = True
    self.inputSelectorGeoDistorsionResult.showHidden = False
    self.inputSelectorGeoDistorsionResult.showChildNodeTypes = False
    self.inputSelectorGeoDistorsionResult.setMRMLScene( slicer.mrmlScene )
    self.inputSelectorGeoDistorsionResult.setToolTip( "Pick the input to the algorithm." )
    parametersFormLayout.addRow("Geometric distorsion result image (only for testing, let it in none): ", self.inputSelectorGeoDistorsionResult)

    #
    # Show 3D Geometric distorsion Button
    #
    self.Show3DGeoDistorsionButton = qt.QPushButton("2B) Show 3D Geometric distorsion results")
    self.Show3DGeoDistorsionButton.toolTip = "Run the Show3D algorithm."
    self.Show3DGeoDistorsionButton.enabled = True
    parametersFormLayout.addRow(self.Show3DGeoDistorsionButton)

    #
    # SNR Button
    #
    self.SNRButton = qt.QPushButton("3) Signal to Noise Ratio analysis")
    self.SNRButton.toolTip = "Run the SNR algorithm."
    self.SNRButton.enabled = True
    parametersFormLayout.addRow(self.SNRButton)


    #
    # Signal Linearity Button
    #
    self.SignalLinearityButton = qt.QPushButton("4) Signal Linearity analysis")
    self.SignalLinearityButton.toolTip = "Run the analysis of the mean signal in the principales spheres."
    self.SignalLinearityButton.enabled = True
    parametersFormLayout.addRow(self.SignalLinearityButton)


    # connections
    self.RegistrationButton.connect('clicked(bool)', self.onRegistrationButton)
    self.SNRButton.connect('clicked(bool)', self.onSNRButton)
    self.SignalLinearityButton.connect('clicked(bool)', self.onSignalLinearityButton)
    self.GeoDistorsionButton.connect('clicked(bool)', self.onGeoDistorsionButton)
    self.Show3DGeoDistorsionButton.connect('clicked(bool)', self.onShow3DGeoDistorsionButton)
    self.inputSelectorCTImage.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    self.inputSelectorMaskImg.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    self.inputSelectorMRImage.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    self.inputSelectorGeoDistorsionResult.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    self.OutputDirectory.connect("Output directory)", self.onSelect)

    # Add vertical spacer
    self.layout.addStretch(1)

    # Refresh regsitration and SNR buttons state
    self.onSelect()

  def cleanup(self):
    pass

  def onSelect(self):
    self.RegistrationButton.enabled = self.inputSelectorCTImage.currentNode() and self.inputSelectorMaskImg.currentNode() and self.inputSelectorMRImage.currentNode()
    self.GeoDistorsionButton.enabled = self.inputSelectorCTImage.currentNode() and self.inputSelectorMaskImg.currentNode() and self.inputSelectorMRImage.currentNode()
    self.Show3DGeoDistorsionButton.enabled = self.inputSelectorCTImage.currentNode()
    self.SNRButton.enabled = self.inputSelectorCTImage.currentNode() and self.inputSelectorMaskImg.currentNode() and self.inputSelectorMRImage.currentNode()
    self.SignalLinearityButton.enabled = self.inputSelectorCTImage.currentNode() and self.inputSelectorMaskImg.currentNode() and self.inputSelectorMRImage.currentNode()

  def onRegistrationButton(self):
    logic = MRI_Quality_ControlLogic()
    logic.recalage(self.inputSelectorCTImage.currentNode(), self.inputSelectorMaskImg.currentNode(), self.inputSelectorMRImage.currentNode(), self.OutputDirectory)

  def onGeoDistorsionButton(self):
    logic2A = MRI_Quality_ControlLogic()
    logic2A.GeoDistorsion(self.inputSelectorCTImage.currentNode(), self.inputSelectorMaskImg.currentNode(), self.OutputDirectory)

  def onShow3DGeoDistorsionButton(self):
    logic2B = MRI_Quality_ControlLogic()
    logic2B.GeoDistorsionShow3D(self.inputSelectorGeoDistorsionResult.currentNode(), self.OutputDirectory)

  def onSNRButton(self):
    logic3 = MRI_Quality_ControlLogic()
    logic3.SNR(self.inputSelectorMaskImg.currentNode(), self.OutputDirectory)

  def onSignalLinearityButton(self):
    logic4 = MRI_Quality_ControlLogic()
    logic4.SignalLinearity(self.inputSelectorMaskImg.currentNode(), self.OutputDirectory)


#
# MRI_Quality_ControlLogic
#

class MRI_Quality_ControlLogic(ScriptedLoadableModuleLogic):

  def recalage(self, inputSelectorCTImage, inputSelectorMaskImg, inputSelectorMRImage, OutputDirectory):
    time1 = time.time()

    ### Fonctions nécessaires pour fonction relacage ci-dessous ###
    def cropImagefctLabel(image, LowerBondingBox, UpperBondingBox ):
      crop=sitk.CropImageFilter()
      image_cropper=crop.Execute(image, LowerBondingBox, UpperBondingBox  )
      return image_cropper

    def recalagerigid_Euler3Dtransform (image_ref, image_mobile, ImageSamplingPercentage, MaskCT, MaskIRM):
      image_ref = sitk.Cast(image_ref, sitk.sitkFloat64) #convertie l'image en float 64
      image_mobile = sitk.Cast(image_mobile, sitk.sitkFloat64) #convertie l'image en float 64
      R = sitk.ImageRegistrationMethod()
      R.SetMetricAsMattesMutualInformation(64) #nombre de bin
      R.SetMetricSamplingPercentage(float(ImageSamplingPercentage)/100, sitk.sitkWallClock)
      R.SetMetricSamplingStrategy(R.RANDOM)
      R.SetOptimizerAsRegularStepGradientDescent(2.0, 0.01, 1500, 0.5 ) #R.SetOptimizerAsRegularStepGradientDescent( maxStep, minStep,numberOfIterations,relaxationFactor );
      tx = sitk.CenteredTransformInitializer(image_ref, image_mobile, sitk.Euler3DTransform())
      R.SetInitialTransform(tx)
      R.SetOptimizerScalesFromIndexShift()
      R.SetInterpolator(sitk.sitkLinear)
      R.SetMetricFixedMask(MaskCT)
      R.SetMetricMovingMask(MaskIRM)
      #R.AddCommand( sitk.sitkIterationEvent, lambda: command_iteration(R) )
      outTx = R.Execute(image_ref, image_mobile)
      InfMut= R.GetMetricValue()
      print(outTx)
      print("    Recalagerigid_Euler3Dtransform: transformation ok")
      return outTx

    def reechantillonage(image_ref,image_mobile, tranformation):
      resampler = sitk.ResampleImageFilter()
      resampler.SetReferenceImage(image_ref)
      resampler.SetInterpolator(sitk.sitkLinear)
      resampler.SetDefaultPixelValue(0)
      resampler.SetTransform(tranformation)
      ImageRecaler = resampler.Execute(image_mobile)
      return ImageRecaler 


    def dilate(image):
      dilateFilter= sitk.BinaryDilateImageFilter()
      dilateFilter.SetKernelRadius(20)
      image_dilated=dilateFilter.Execute(image)
      return image_dilated


    def separate_label (label_template):
      Connected=sitk.ConnectedComponentImageFilter()
      connectedlabel=Connected.Execute( label_template, True)
      Relabel=sitk.RelabelComponentImageFilter()
      connectedlabel=Relabel.Execute(connectedlabel, 10 ,True)# trie par taille
      #su.PushToSlicer(connectedlabel, "connectedlabel", 2)
      return connectedlabel
      

    def multires_registration(fixed_image, moving_image, initial_transform, ImageSamplingPercentage):
      registration_method = sitk.ImageRegistrationMethod()
      registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
      registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
      registration_method.SetMetricSamplingPercentage(float(ImageSamplingPercentage)/100)
      registration_method.SetInterpolator(sitk.sitkLinear)
      registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100, estimateLearningRate=registration_method.EachIteration) #Once
      registration_method.SetOptimizerScalesFromPhysicalShift() 
      registration_method.SetInitialTransform(initial_transform)
      registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
      registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas = [2,1,0])
      registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

      final_transform = registration_method.Execute(fixed_image, moving_image)
      print('Final metric value: {0}'.format(registration_method.GetMetricValue()))
      print('Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))
      return final_transform

    ###### Récupération des images et dossier de sauvegarde sélectionnés par l'utilisateur ####
    print("Début de la fonction recalage")
    nom_image=inputSelectorMRImage
    nom_image_ref=inputSelectorCTImage
    nom_template=inputSelectorMaskImg

    ######################importer les images##################
    image_IRM=su.PullVolumeFromSlicer(nom_image)
    image_CT=su.PullVolumeFromSlicer(nom_image_ref)
    label_CT=su.PullVolumeFromSlicer(nom_template) #dans l'espace CT
    #label_CT_dilated=su.PullVolumeFromSlicer(nom_template2)
    #image_ideal= sitk.Cast(image_ideal, sitk.sitkFloat64) #convertie l'image en float 64 
    #print time.ctime() 
    print("Importation:ok")
    ###########################vrecalage en 3 étape##################
    image_CT = sitk.Cast(image_CT, sitk.sitkFloat64)
    image_IRM = sitk.Cast(image_IRM, sitk.sitkFloat64)
    ####################recalage primaire de la sommes des moments des images###########
    initial_transform = sitk.CenteredTransformInitializer(image_CT, image_IRM, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.MOMENTS)
    #####################recalage multiresolution rigide des images #################################
    medium_transform =multires_registration(image_CT, image_IRM, initial_transform, 5)
    #############################recalage multiresolution du centre du phantome moins soumis aux distortions geom#######################
    ####crop sur les 5 plus grosse sphéres#######################
    mask_label=sitk.BinaryThreshold(label_CT, 250, 254, 1, 0) #creer un masque sur les 5 plus grosse spheres
    stats= sitk.LabelIntensityStatisticsImageFilter()
    stats.Execute(mask_label, image_CT)
    delta=10 #additionnal voxel to crop volume to avoid border problem
    LowerBondingBox=[stats.GetBoundingBox(1)[0]-delta,stats.GetBoundingBox(1)[1]-delta,stats.GetBoundingBox(1)[2]-delta]
    UpperBondingBox=[image_CT.GetSize()[0]-(stats.GetBoundingBox(1)[0]+stats.GetBoundingBox(1)[3]+delta),image_CT.GetSize()[1]-(stats.GetBoundingBox(1)[1]+stats.GetBoundingBox(1)[4]+delta),image_CT.GetSize()[2]-(stats.GetBoundingBox(1)[2]+stats.GetBoundingBox(1)[5]+delta)]   
    image_CT_crop=cropImagefctLabel(image_CT, LowerBondingBox, UpperBondingBox ) 
    final_transform =multires_registration(image_CT_crop, image_IRM, medium_transform, 20)
    ##################reechantillonage############
    image_IRM_reg=reechantillonage(image_CT,image_IRM, final_transform)     
    su.PushToSlicer(image_IRM_reg, "verif_image_IRMrecaler",1) 
    

    time2 = time.time()
    TimeForrunFunction = time2 - time1
    print("\n")
    print(u"La fonction recalage s'est bien executée (temps = " + str(TimeForrunFunction) +" secondes)")
   

  def GeoDistorsion(self, inputSelectorCTImage, inputSelectorMaskImg, OutputDirectory):
    #### Fonction servant à calculer la distorsion géométrique de l'IRM en comparant la distance des sphères du fantôme en CT et en IRM ###
    #### Output : fichier excel avec les différences entre les centroids des sphères en CT et en IRM ###
 
    def creation_LabelAvecVolumeSphereConnu_OTSU(image,label_CT):   
      mask_label=sitk.BinaryThreshold(label_CT, 1, 200, 1, 0) #creer un masque sur toute les sphéres
      Otsu = sitk.OtsuThresholdImageFilter() 
      labelOtsu=Otsu.Execute(image, mask_label, 0, 1, 64, True, 1) #otsu sur l'ensemble sde sphéres-> label value=1
      label_CT= sitk.Cast(label_CT, sitk.sitkUInt8)
      labelOtsu= sitk.Cast(labelOtsu, sitk.sitkUInt8)   
      label_ideal=sitk.Multiply(label_CT,labelOtsu) #multplie par label initiaux pour retrouver leur valeur
      label_ideal= sitk.Cast(label_ideal, sitk.sitkUInt8) #transforme pour format de label
      print("    creation_LabelAvecVolumeSphereConnu: recherche du seuil optimun:ok")
      su.PushToSlicer(label_ideal, "label_ideal",2) 
      return label_ideal

    def DeformInColor(LabelVolumeSphere_IRM, distance, Nlabels) :
      Add=sitk.AddImageFilter()
      NSpheresTolerance= np.zeros(3)
      image_distance=sitk.BinaryThreshold(LabelVolumeSphere_IRM, 500, 500, 0,0)
      for i in Nlabels:
          if distance[i]<=1.0:
              image_distance=Add.Execute(image_distance,sitk.BinaryThreshold(LabelVolumeSphere_IRM, i, i, 6,0))
              NSpheresTolerance[0]+=1
          elif distance[i]<=2.0:
              image_distance=Add.Execute(image_distance,sitk.BinaryThreshold(LabelVolumeSphere_IRM, i, i, 13,0))
              NSpheresTolerance[1]+=1
          else :
              image_distance=Add.Execute(image_distance,sitk.BinaryThreshold(LabelVolumeSphere_IRM, i, i, 14,0))
              NSpheresTolerance[2]+=1
      su.PushToSlicer(image_distance, "Resultat_DisGeo_CT_MRI" + str(date.day) + str(date.month) + str(date.year), 2)
      return NSpheresTolerance

    print("Début de la fonction geometric distorsion")
    time1 = time.time()

    ###### Récupération des images d'intérêt ####
    image_IRM_reg = su.PullVolumeFromSlicer("verif_image_IRMrecaler")
    nom_template=inputSelectorMaskImg
    label_CT=su.PullVolumeFromSlicer(nom_template)
    nom_image_ref=inputSelectorCTImage
    image_CT=su.PullVolumeFromSlicer(nom_image_ref)
    savepath = OutputDirectory.directory + "/Analyse_QC_MRI" + str(date.day) + str(date.month)  + str(date.year) + ".txt"

    ########################segmentation des spheresIRM #####################    
    LabelVolumeSphere_IRM=creation_LabelAvecVolumeSphereConnu_OTSU(image_IRM_reg, label_CT) #max 200 sphéres
 

    ###Analyse######################################################
    stat_filter_CT=sitk.LabelIntensityStatisticsImageFilter()
    stat_filter_IRM=sitk.LabelIntensityStatisticsImageFilter()
    stat_filter_CT.Execute(label_CT, image_CT) #attention à l'ordre
    stat_filter_IRM.Execute(LabelVolumeSphere_IRM, image_IRM_reg) #attention à l'ordre
    Nlabels=stat_filter_IRM.GetLabels()
    distance=np.zeros(max(Nlabels)+1)
    
    #############################test si le volume segmenter est cohérent (problème de fuite de solution...) sinon l'élinime:
    ErreurSurVolume=0.5 # erreur inclusion de la segmentation de 50%
    t=0
    list_label=list(Nlabels)
    for i in Nlabels:
        if ( stat_filter_IRM.GetEquivalentSphericalRadius(i)<5*(1-ErreurSurVolume) ) or ( stat_filter_IRM.GetEquivalentSphericalRadius(i)>5*(1+ErreurSurVolume) ): #tolérance pour sphére de rayon 5mm
            list_label.remove(i) #☻cette fonction ne marche pas avec les tuple (immutable) mais avec les list
            t=t+1

    Nlabels=tuple(list_label)
    print ("Nombre de sphere éléminées si Rspheres different de +/- 50%: ")
    print( t )

    ###################################
    ###Boucle ecriture du fichier csv de sortie###
    f = open(savepath, 'a')
    f.write( str(time.ctime()) )
    f.write("\n")
    f.write(str("label \t x_CT \t y_CT \t z_CT \t  x_IRM \t y_IRM \t z_IRM \t distance"))
    f.write("\n")
    X_spacing=image_CT.GetSpacing()[0]
    Y_spacing=image_CT.GetSpacing()[1]
    Z_spacing=image_CT.GetSpacing()[2]
    for i in Nlabels:
        x_CT=stat_filter_CT.GetCentroid(i)[0]
        y_CT=stat_filter_CT.GetCentroid(i)[1]
        z_CT=stat_filter_CT.GetCentroid(i)[2]
        x_IRM=stat_filter_IRM.GetCentroid(i)[0]
        y_IRM=stat_filter_IRM.GetCentroid(i)[1]
        z_IRM=stat_filter_IRM.GetCentroid(i)[2]
        f.write(str(i))
        f.write("\t")
        f.write(str(x_CT))
        f.write("\t")
        f.write(str(y_CT))
        f.write("\t")
        f.write(str(z_CT)) 
        f.write("\t")
        f.write(str(x_IRM))
        f.write("\t")
        f.write(str(y_IRM))
        f.write("\t")
        f.write(str(z_IRM))
        f.write("\t")
        distance[i]=sqrt(((x_CT-x_IRM)*X_spacing)**2+((y_CT-y_IRM)*Y_spacing)**2+((z_CT-z_IRM)*Z_spacing)**2)
        f.write(str(distance[i]))
        f.write("\n")                  
    print(u"Ecriture du fichier résultat ok")

    ##################################creation de l'image de deformation############
    ResultatTolerance=DeformInColor(LabelVolumeSphere_IRM, distance, Nlabels)
    f.write("\n")
    f.write("\n")
    f.write(str("N sphere <1mm \t N sphere <2mm \t N sphere >2mm "))
    f.write("\n")
    f.write(str(ResultatTolerance[0])+"\t"+str(ResultatTolerance[1])+"\t"+str(ResultatTolerance[2]) ) 
    f.write("\n")
    f.write("\n")
    f.close()
    print("N sphere <1mm:")
    print(ResultatTolerance[0])
    print("N sphere <2mm:")
    print(ResultatTolerance[1])
    print("N sphere >2mm:")
    print(ResultatTolerance[2])
    print("main : creation du label avec contrainte couleur ok")
    time2 = time.time()
    TimeForrunFunction = time2 - time1
    print("\n")
    print(u"L'analyse de distorsion géométrique s'est bien executée (temps = " + str(TimeForrunFunction) +" secondes)")
    print("\n")
    print(u"Pour une représentation 3D des sphères et de la tolérance quant à la déformation géométrique observée, cliquez sur Show3D")

  def GeoDistorsionShow3D(self, inputSelectorGeoDistorsionResult, OutputDirectory):
    ### Output : image 3D avec code couleur en fonction du fait que les sphères ont des déformations géométrique en dessus ou en dessous des valeurs de tolérance ####
    print(u"Début de la fonction Show3D")
    time1 = time.time()

    if not inputSelectorGeoDistorsionResult:
      nom_image_IRMrecal = "Resultat_DisGeo_CT_MRI"+ str(date.day) + str(date.month)  + str(date.year)
    else:
      nom_image_IRMrecal=inputSelectorGeoDistorsionResult

    #############   Récupération du chemin de sauvegarde de l'image résultat (3D) #############
    ThreeDImagesavepath = OutputDirectory.directory + "/3DImageResult" + str(date.day) + str(date.month)  + str(date.year) +".png"
    
    
    #############  Récupération de l'image résultat des test précédent et création de 3 volumes pour les 3 conditions de tolérance  #############
    volumeNode1 = slicer.util.getNode(nom_image_IRMrecal)
    labelVolumeNode1 = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
    slicer.vtkSlicerVolumesLogic().CreateLabelVolumeFromVolume(slicer.mrmlScene, labelVolumeNode1, volumeNode1)

    volumeNode2 = slicer.util.getNode(nom_image_IRMrecal)
    labelVolumeNode2 = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
    slicer.vtkSlicerVolumesLogic().CreateLabelVolumeFromVolume(slicer.mrmlScene, labelVolumeNode2, volumeNode2)

    volumeNode3 = slicer.util.getNode(nom_image_IRMrecal)
    labelVolumeNode3 = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
    slicer.vtkSlicerVolumesLogic().CreateLabelVolumeFromVolume(slicer.mrmlScene, labelVolumeNode3, volumeNode3)


    ############# Création de 3 segmentations basée sur les valeurs de tolérance 6; 13 et 14 du test précédent  #############
    thresholdValue1 = 6
    voxelArray = slicer.util.arrayFromVolume(volumeNode1)
    labelVoxelArray = slicer.util.arrayFromVolume(labelVolumeNode1)
    labelVoxelArray[voxelArray == thresholdValue1] = 100
    labelVoxelArray[voxelArray != thresholdValue1] = 0
    slicer.util.arrayFromVolumeModified(labelVolumeNode1)

    thresholdValue2 = 13
    voxelArray = slicer.util.arrayFromVolume(volumeNode2)
    labelVoxelArray = slicer.util.arrayFromVolume(labelVolumeNode2)
    labelVoxelArray[voxelArray == thresholdValue2] = 100
    labelVoxelArray[voxelArray != thresholdValue2] = 0
    slicer.util.arrayFromVolumeModified(labelVolumeNode2)

    thresholdValue3 = 14
    voxelArray = slicer.util.arrayFromVolume(volumeNode3)
    labelVoxelArray = slicer.util.arrayFromVolume(labelVolumeNode3)
    labelVoxelArray[voxelArray == thresholdValue3] = 100
    labelVoxelArray[voxelArray != thresholdValue3] = 0
    slicer.util.arrayFromVolumeModified(labelVolumeNode3)


    ############# Création de 3 segments issus des 3 segmentations précédentes, coloriage en fonction de la tolérance ou non au test #############
    segmentationNode1 = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode')
    segmentID = slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labelVolumeNode1, segmentationNode1)
    segmentation = segmentationNode1.GetSegmentation()
    segment = segmentation.GetSegment(segmentation.GetNthSegmentID(0))
    segment.SetColor(0,1,0)

    segmentationNode2 = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode')
    slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labelVolumeNode2, segmentationNode2)
    segmentation = segmentationNode2.GetSegmentation()
    segment = segmentation.GetSegment(segmentation.GetNthSegmentID(0))
    try: # condition pour éviter plantage dans le cas où il n'y ait pas de sphères avec des distortions < 2mm
        segment.SetColor(1,0.70,0) # couleur R, V, B
    except:
        print("Il n'y a pas de sphère avec une distortion inférieur à 2mm")
        sphere2exist = False
    else:
        sphere2exist = True

    segmentationNode3 = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode')
    slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labelVolumeNode3, segmentationNode3)
    segmentation = segmentationNode3.GetSegmentation()
    segment = segmentation.GetSegment(segmentation.GetNthSegmentID(0))
    try: # condition pour éviter plantage dans le cas où il n'y ait pas de sphères avec des distortions > 2mm
        segment.SetColor(1,0,0) # couleur R, V, B 
    except:
        print("Il n'y a pas de sphère avec une distortion supérieur à 2mm")
        sphere3exist = False
    else:
        sphere3exist = True


    ############# Représentation 3d des volumes segmentés et coloriés   #############
    segmentationNode1.CreateClosedSurfaceRepresentation()
    if sphere2exist == True:
        segmentationNode2.CreateClosedSurfaceRepresentation()
    if sphere3exist == True:
        segmentationNode3.CreateClosedSurfaceRepresentation()


    ############# Copie d'écran du résultat et sauvegarde de l'image dans le dossier de résultat global  #############
    ThreeDImage = qt.QPixmap.grabWidget(slicer.util.mainWindow()).toImage()
    ThreeDImage.save(ThreeDImagesavepath)
    
    time2 = time.time()
    TimeForrunFunction = time2 - time1

    print("\n")
    print(u"La fonction de représentation 3D des sphères s'est bien executée (temps = " + str(TimeForrunFunction) +" secondes)")
    print("\n")
    print(u"Si dans la représentation 3D des sphères sont de couleurs rouges, merci de prévenir Cyril Jaudet au 5690 ou Aurélien Corroyer-Dulmont au 5768")
    print("\n")
    print(u"Si besoin d’aide ou d’informations sur comment fonctionne le programme : Cyril Jaudet (5690) ou Aurélien Corroyer-Dulmont (5768)")

  def SNR(self, inputSelectorMaskImg, OutputDirectory):
    #### Fonction servant à calculer la rapport signal sur bruit entre les sphères d'intérêt et le reste du fantôme ainsi que par rapport au bruit ###
    time1 = time.time()
    
    ### Récupération des images d'intérêt ###
    nom_template=inputSelectorMaskImg
    label_Of_Interest=su.PullVolumeFromSlicer(nom_template)
    IRM_recal = su.PullVolumeFromSlicer('verif_image_IRMrecaler')
    savepath = OutputDirectory.directory + "/Analyse_QC_MRI" + str(date.day) + str(date.month)  + str(date.year) + ".txt"
    
    ### Création des filtres ITK pour obtention des données ###
    stat_filter_label_Of_Interest=sitk.LabelIntensityStatisticsImageFilter()
    stat_filter_label_Of_Interest.Execute(label_Of_Interest, IRM_recal)

    ### Analyse des données dans les différents labels d'intérêt ###
    MeanInside = stat_filter_label_Of_Interest.GetMean(255)
    VarianceInside = stat_filter_label_Of_Interest.GetVariance(255)
    SDInside = sqrt(VarianceInside)
    MeanOutside = stat_filter_label_Of_Interest.GetMean(256)
    VarianceOutside = stat_filter_label_Of_Interest.GetVariance(256)
    SDOutside = sqrt(VarianceOutside)

    Mean_Label_Of_Interest = []
    Variance_Label_Of_Interest = []

    for i in range(250,255):
        temp_Mean_Label_Of_Interest = stat_filter_label_Of_Interest.GetMean(i)
        Mean_Label_Of_Interest.append(temp_Mean_Label_Of_Interest)
        temp_Variance_Label_Of_Interest = stat_filter_label_Of_Interest.GetVariance(i)
        Variance_Label_Of_Interest.append(temp_Variance_Label_Of_Interest)

    Mean_Label_Of_Interest = np.mean(Mean_Label_Of_Interest)
    Variance_Label_Of_Interest = np.mean(Variance_Label_Of_Interest)

    SD_Label_Of_Interest = sqrt(Variance_Label_Of_Interest)

    SNRInsidePhantom = float(Mean_Label_Of_Interest) / float(SDInside)
    SNROutsidePhantom = float(Mean_Label_Of_Interest) / float(SDOutside)
    
    SNRInsidePhantom = SNRInsidePhantom/100
    SNROutsidePhantom = SNROutsidePhantom/100

    ### Enoncé des résultats ###
    print("\n")
    print(u"Moyenne des labels :\nSphères : " + str(Mean_Label_Of_Interest) + u"  Intérieur du fantôme : " + str(MeanInside) + u"  Extérieur du fantôme : " + str(MeanOutside))
    print("\n")
    print(u"Variance des labels :\nSphères : " + str(Variance_Label_Of_Interest) + u"  Intérieur du fantôme : " + str(VarianceInside) + u"  Extérieur du fantôme : " + str(VarianceOutside))
    print("\n")
    print(u"Ecart-type des labels :\nSphères : " + str(SD_Label_Of_Interest) + u"  Intérieur du fantôme : " + str(SDInside) + u"  Extérieur du fantôme : " + str(SDOutside))
    print("\n")
    print(u"La valeur du SNR à l'intérieur du fantôme est de : " + str(SNRInsidePhantom) + " %")
    print("\n")
    print(u"La valeur du SNR à l'extérieur du fantôme est de : " + str(SNROutsidePhantom) + " %")

    ### Ecriture des résultats dans le fichier d'analyse.csv ###
    f = open(savepath, 'a')

    ### encodage du fichier pour écriture incluant les "é" ###
    f = codecs.open(savepath, "a", encoding='Latin-1')

    f.write("\n\n\n")
    f.write(u"Résultats de l'analyse du SNR:\n\n")
    f.write(u"Moyenne des labels :\nSphères : " + str(Mean_Label_Of_Interest) + u"  Intérieur du fantôme : " + str(MeanInside) + u"  Extérieur du fantôme : " + str(MeanOutside))
    f.write("\n\n")
    f.write(u"Variance des labels :\nSphères : " + str(Variance_Label_Of_Interest) + u"  Intérieur du fantôme : " + str(VarianceInside) + u"  Extérieur du fantôme : " + str(VarianceOutside))
    f.write("\n\n")
    f.write(u"Ecart-type des labels :\nSphères : " + str(SD_Label_Of_Interest) + u"  Intérieur du fantôme : " + str(SDInside) + u"  Extérieur du fantôme : " + str(SDOutside))
    f.write("\n\n")
    f.write(u"La valeur du SNR à l'intérieur du fantôme est de : " + str(SNRInsidePhantom))
    f.write("\n\n")
    f.write(u"La valeur du SNR à l'extérieur du fantôme est de : " + str(SNROutsidePhantom))
    f.close()

    time2 = time.time()
    TimeForrunFunction = time2 - time1
    print("\n")
    print(u"La fonction SNR s'est executée en " + str(TimeForrunFunction) +" secondes")


  def SignalLinearity(self, inputSelectorMaskImg, OutputDirectory):
    time1 = time.time()
    
    ### Récupération des images d'intérêt ###
    nom_template=inputSelectorMaskImg
    label_Of_Interest=su.PullVolumeFromSlicer(nom_template)
    IRM_recal = su.PullVolumeFromSlicer('verif_image_IRMrecaler')
    savepath = OutputDirectory.directory + "/Analyse_QC_MRI" + str(date.day) + str(date.month)  + str(date.year) + ".txt"
    
    ### Création des filtres ITK pour obtention des données ###
    stat_filter_label_Of_Interest=sitk.LabelIntensityStatisticsImageFilter()
    stat_filter_label_Of_Interest.Execute(label_Of_Interest, IRM_recal)

    Mean_Label_Of_Interest = []
    Variance_Label_Of_Interest = []

    for i in range(250,255):
      Mean_Label_Of_Interest.append(stat_filter_label_Of_Interest.GetMean(i))

    ### Enoncé des résultats ###
    print("\n")
    print(u"Signal moyen de la sphère centrale : " + str(Mean_Label_Of_Interest[0]))
    print("\n")
    print(u"Signal moyen de la sphère moyenne n°1 (bas/gauche) : " + str(Mean_Label_Of_Interest[2]))
    print("\n")
    print(u"Signal moyen de la sphère moyenne n°2 (bas/droit) : " + str(Mean_Label_Of_Interest[4]))
    print("\n")
    print(u"Signal moyen de la sphère moyenne n°3 (haut/droit) : " + str(Mean_Label_Of_Interest[3]))
    print("\n")
    print(u"Signal moyen de la sphère moyenne n°4 (haut/gauche) : " + str(Mean_Label_Of_Interest[1]))

    ### Ecriture des résultats dans le fichier d'analyse.csv ###
    f = open(savepath, 'a')

    ### encodage du fichier pour écriture incluant les "é" ###
    f = codecs.open(savepath, "a", encoding='Latin-1')

    f.write("\n\n\n")
    f.write(u"Résultats de l'analyse de la linéarité du signal :\n\n")
    f.write(u"Signal moyen de la sphère centrale : " + str(Mean_Label_Of_Interest[0]))
    f.write("\n\n")
    f.write(u"Signal moyen de la sphère moyenne n°1 (bas/gauche) : " + str(Mean_Label_Of_Interest[2]))
    f.write("\n\n")
    f.write(u"Signal moyen de la sphère moyenne n°2 (bas/droit) : " + str(Mean_Label_Of_Interest[4]))
    f.write("\n\n")
    f.write(u"Signal moyen de la sphère moyenne n°3 (haut/droit) : " + str(Mean_Label_Of_Interest[3]))
    f.write("\n\n")
    f.write(u"Signal moyen de la sphère moyenne n°4 (haut/gauche) : " + str(Mean_Label_Of_Interest[1]))
    f.close()

    time2 = time.time()
    TimeForrunFunction = time2 - time1
    print("\n")
    print(u"La fonction Signal Linearity s'est executée en " + str(TimeForrunFunction) +" secondes")    
