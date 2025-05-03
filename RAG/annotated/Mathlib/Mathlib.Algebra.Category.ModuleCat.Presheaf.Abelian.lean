noncomputable instance : NormalEpiCategory (PresheafOfModules.{v} R) where
  normalEpiOfEpi p _ := NormalEpi.mk _ (kernel.ι p) (kernel.condition _)
    (evaluationJointlyReflectsColimits _ _ (fun _ =>
      Abelian.isColimitMapCoconeOfCokernelCoforkOfπ _ _))


noncomputable instance : NormalMonoCategory (PresheafOfModules.{v} R) where
  normalMonoOfMono i _ := NormalMono.mk _ (cokernel.π i) (cokernel.condition _)
    (evaluationJointlyReflectsLimits _ _ (fun _ =>
      Abelian.isLimitMapConeOfKernelForkOfι _ _))


noncomputable instance : Abelian (PresheafOfModules.{v} R) where


