/-- A right homology data for a short complex `S` consists of morphisms `p : S.X₂ ⟶ Q` and
`ι : H ⟶ Q` such that `p` identifies `Q` to the kernel of `f : S.X₁ ⟶ S.X₂`,
and that `ι` identifies `H` to the kernel of the induced map `g' : Q ⟶ S.X₃` -/
structure RightHomologyData where
  /-- a choice of cokernel of `S.f : S.X₁ ⟶ S.X₂`-/
  Q : C
  /-- a choice of kernel of the induced morphism `S.g' : S.Q ⟶ X₃`-/
  H : C
  /-- the projection from `S.X₂` -/
  p : S.X₂ ⟶ Q
  /-- the inclusion of the (right) homology in the chosen cokernel of `S.f` -/
  ι : H ⟶ Q
  /-- the cokernel condition for `p` -/
  wp : S.f ≫ p = 0
  /-- `p : S.X₂ ⟶ Q` is a cokernel of `S.f : S.X₁ ⟶ S.X₂` -/
  hp : IsColimit (CokernelCofork.ofπ p wp)
  /-- the kernel condition for `ι` -/
  wι : ι ≫ hp.desc (CokernelCofork.ofπ _ S.zero) = 0
  /-- `ι : H ⟶ Q` is a kernel of `S.g' : Q ⟶ S.X₃` -/
  hι : IsLimit (KernelFork.ofι ι wι)


/-- The chosen cokernels and kernels of the limits API give a `RightHomologyData` -/
@[simps]
noncomputable def ofHasCokernelOfHasKernel
    [HasCokernel S.f] [HasKernel (cokernel.desc S.f S.g S.zero)] :
    S.RightHomologyData :=
{ Q := cokernel S.f,
  H := kernel (cokernel.desc S.f S.g S.zero),
  p := cokernel.π _,
  ι := kernel.ι _,
  wp := cokernel.condition _,
  hp := cokernelIsCokernel _,
  wι := kernel.condition _,
  hι := kernelIsKernel _, }


attribute [reassoc (attr := simp)] wp wι


instance : Epi h.p := ⟨fun _ _ => Cofork.IsColimit.hom_ext h.hp⟩


instance : Mono h.ι := ⟨fun _ _ => Fork.IsLimit.hom_ext h.hι⟩


/-- Any morphism `k : S.X₂ ⟶ A` such that `S.f ≫ k = 0` descends
to a morphism `Q ⟶ A` -/
def descQ (k : S.X₂ ⟶ A) (hk : S.f ≫ k = 0) : h.Q ⟶ A :=
  h.hp.desc (CokernelCofork.ofπ k hk)


@[reassoc (attr := simp)]
lemma p_descQ (k : S.X₂ ⟶ A) (hk : S.f ≫ k = 0) : h.p ≫ h.descQ k hk = k :=
  h.hp.fac _ WalkingParallelPair.one


/-- The morphism from the (right) homology attached to a morphism
`k : S.X₂ ⟶ A` such that `S.f ≫ k = 0`. -/
@[simp]
def descH (k : S.X₂ ⟶ A) (hk : S.f ≫ k = 0) : h.H ⟶ A :=
  h.ι ≫ h.descQ k hk


/-- The morphism `h.Q ⟶ S.X₃` induced by `S.g : S.X₂ ⟶ S.X₃` and the fact that
`h.Q` is a cokernel of `S.f : S.X₁ ⟶ S.X₂`. -/
def g' : h.Q ⟶ S.X₃ := h.descQ S.g S.zero


@[reassoc (attr := simp)] lemma p_g' : h.p ≫ h.g' = S.g := p_descQ _ _ _


@[reassoc (attr := simp)] lemma ι_g' : h.ι ≫ h.g' = 0 := h.wι


@[reassoc]
lemma ι_descQ_eq_zero_of_boundary (k : S.X₂ ⟶ A) (x : S.X₃ ⟶ A) (hx : k = S.g ≫ x) :
                        /-
                          C : Type u_1
                          inst✝¹ : CategoryTheory.Category.{?u.11893, u_1} C
                          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                          S S₁ S₂ S₃ : CategoryTheory.ShortComplex C
                          h : S.RightHomologyData
                          A : C
                          k : Quiver.Hom S.X₂ A
                          x : Quiver.Hom S.X₃ A
                          hx : Eq k (CategoryTheory.CategoryStruct.comp S.g x)
                          ⊢ Eq (CategoryTheory.CategoryStruct.comp S.f k) 0
                        -/
    h.ι ≫ h.descQ k (by rw [hx, S.zero_assoc, zero_comp]) = 0 := by
                        /-
                          🎉 no goals
                        -/
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    h : S.RightHomologyData
    A : C
    k : Quiver.Hom S.X₂ A
    x : Quiver.Hom S.X₃ A
    hx : Eq k (CategoryTheory.CategoryStruct.comp S.g x)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp h.ι (h.descQ k ⋯)) 0
  -/
  rw [show 0 = h.ι ≫ h.g' ≫ x by simp]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    h : S.RightHomologyData
    A : C
    k : Quiver.Hom S.X₂ A
    x : Quiver.Hom S.X₃ A
    hx : Eq k (CategoryTheory.CategoryStruct.comp S.g x)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp h.ι (h.descQ k ⋯)) (CategoryTheory.Ca …
  -/
  congr 1
  /-
    case e_a
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    h : S.RightHomologyData
    A : C
    k : Quiver.Hom S.X₂ A
    x : Quiver.Hom S.X₃ A
    hx : Eq k (CategoryTheory.CategoryStruct.comp S.g x)
    ⊢ Eq (h.descQ k ⋯) (CategoryTheory.CategoryStruct.comp h.g' x)
  -/
  simp only [← cancel_epi h.p, hx, p_descQ, p_g'_assoc]
  /-
    🎉 no goals
  -/


/-- For `h : S.RightHomologyData`, this is a restatement of `h.hι`, saying that
`ι : h.H ⟶ h.Q` is a kernel of `h.g' : h.Q ⟶ S.X₃`. -/
def hι' : IsLimit (KernelFork.ofι h.ι h.ι_g') := h.hι


/-- The morphism `A ⟶ H` induced by a morphism `k : A ⟶ Q` such that `k ≫ g' = 0` -/
def liftH (k : A ⟶ h.Q) (hk : k ≫ h.g' = 0) : A ⟶ h.H :=
  h.hι.lift (KernelFork.ofι k hk)


@[reassoc (attr := simp)]
lemma liftH_ι (k : A ⟶ h.Q) (hk : k ≫ h.g' = 0) : h.liftH k hk ≫ h.ι = k :=
  h.hι.fac (KernelFork.ofι k hk) WalkingParallelPair.zero


lemma isIso_p (hf : S.f = 0) : IsIso h.p :=
                        /-
                          C : Type u_1
                          inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
                          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                          S : CategoryTheory.ShortComplex C
                          h : S.RightHomologyData
                          hf : Eq S.f 0
                          ⊢ Eq (CategoryTheory.CategoryStruct.comp S.f (CategoryTheory.CategoryStruct.id …
                        -/
  ⟨h.descQ (𝟙 S.X₂) (by rw [hf, comp_id]), p_descQ _ _ _, by
                        /-
                          🎉 no goals
                        -/
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      S : CategoryTheory.ShortComplex C
      h : S.RightHomologyData
      hf : Eq S.f 0
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (h.descQ (CategoryTheory.CategoryStru …
    -/
    simp only [← cancel_epi h.p, p_descQ_assoc, id_comp, comp_id]⟩
    /-
      🎉 no goals
    -/


lemma isIso_ι (hg : S.g = 0) : IsIso h.ι := by
  have ⟨φ, hφ⟩ := KernelFork.IsLimit.lift' h.hι' (𝟙 _)
    (by rw [← cancel_epi h.p, id_comp, p_g', comp_zero, hg])
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    h : S.RightHomologyData
    hg : Eq S.g 0
    φ : Quiver.Hom h.Q (CategoryTheory.Limits.KernelFork.ofι h.ι ⋯).pt
    hφ : Eq (CategoryTheory.CategoryStruct.comp φ (CategoryTheory.Limits.Fork.ι (C …
    ⊢ CategoryTheory.IsIso h.ι
  -/
  dsimp at hφ
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    h : S.RightHomologyData
    hg : Eq S.g 0
    φ : Quiver.Hom h.Q (CategoryTheory.Limits.KernelFork.ofι h.ι ⋯).pt
    hφ : Eq (CategoryTheory.CategoryStruct.comp φ h.ι) (CategoryTheory.CategoryStr …
    ⊢ CategoryTheory.IsIso h.ι
  -/
  exact ⟨φ, by rw [← cancel_mono h.ι, assoc, hφ, comp_id, id_comp], hφ⟩
  /-
    🎉 no goals
  -/


/-- When the first map `S.f` is zero, this is the right homology data on `S` given
by any limit kernel fork of `S.g` -/
@[simps]
def ofIsLimitKernelFork (hf : S.f = 0) (c : KernelFork S.g) (hc : IsLimit c) :
    S.RightHomologyData where
  Q := S.X₂
  H := c.pt
  p := 𝟙 _
  ι := c.ι
           /-
             C : Type u_1
             inst✝¹ : CategoryTheory.Category.{?u.20254, u_1} C
             inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
             S S₁ S₂ S₃ : CategoryTheory.ShortComplex C
             h : S.RightHomologyData
             A : C
             hf : Eq S.f 0
             c : CategoryTheory.Limits.KernelFork S.g
             hc : CategoryTheory.Limits.IsLimit c
             ⊢ Eq (CategoryTheory.CategoryStruct.comp S.f (CategoryTheory.CategoryStruct.id …
           -/
  wp := by rw [comp_id, hf]
           /-
             🎉 no goals
           -/
  hp := CokernelCofork.IsColimit.ofId _ hf
  wι := KernelFork.condition _
                                                         /-
                                                           C : Type u_1
                                                           inst✝¹ : CategoryTheory.Category.{?u.20254, u_1} C
                                                           inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                                           S S₁ S₂ S₃ : CategoryTheory.ShortComplex C
                                                           h : S.RightHomologyData
                                                           A : C
                                                           hf : Eq S.f 0
                                                           c : CategoryTheory.Limits.KernelFork S.g
                                                           hc : CategoryTheory.Limits.IsLimit c
                                                           ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Iso.refl c.pt).hom (C …
                                                         -/
  hι := IsLimit.ofIsoLimit hc (Fork.ext (Iso.refl _) (by aesop_cat))
                                                         /-
                                                           🎉 no goals
                                                         -/


@[simp] lemma ofIsLimitKernelFork_g' (hf : S.f = 0) (c : KernelFork S.g)
    (hc : IsLimit c) : (ofIsLimitKernelFork S hf c hc).g' = S.g := by
  rw [← cancel_epi (ofIsLimitKernelFork S hf c hc).p, p_g',
    ofIsLimitKernelFork_p, id_comp]


/-- When the first map `S.f` is zero, this is the right homology data on `S` given by
the chosen `kernel S.g` -/
@[simps!]
noncomputable def ofHasKernel [HasKernel S.g] (hf : S.f = 0) : S.RightHomologyData :=
ofIsLimitKernelFork S hf _ (kernelIsKernel _)


/-- When the second map `S.g` is zero, this is the right homology data on `S` given
by any colimit cokernel cofork of `S.g` -/
@[simps]
def ofIsColimitCokernelCofork (hg : S.g = 0) (c : CokernelCofork S.f) (hc : IsColimit c) :
    S.RightHomologyData where
  Q := c.pt
  H := c.pt
  p := c.π
  ι := 𝟙 _
  wp := CokernelCofork.condition _
                                                               /-
                                                                 C : Type u_1
                                                                 inst✝¹ : CategoryTheory.Category.{?u.25155, u_1} C
                                                                 inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                                                 S S₁ S₂ S₃ : CategoryTheory.ShortComplex C
                                                                 h : S.RightHomologyData
                                                                 A : C
                                                                 hg : Eq S.g 0
                                                                 c : CategoryTheory.Limits.CokernelCofork S.f
                                                                 hc : CategoryTheory.Limits.IsColimit c
                                                                 ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π c) (C …
                                                               -/
  hp := IsColimit.ofIsoColimit hc (Cofork.ext (Iso.refl _) (by aesop_cat))
                                                               /-
                                                                 🎉 no goals
                                                               -/
                                        /-
                                          C : Type u_1
                                          inst✝¹ : CategoryTheory.Category.{?u.25155, u_1} C
                                          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                          S S₁ S₂ S₃ : CategoryTheory.ShortComplex C
                                          h : S.RightHomologyData
                                          A : C
                                          hg : Eq S.g 0
                                          c : CategoryTheory.Limits.CokernelCofork S.f
                                          hc : CategoryTheory.Limits.IsColimit c
                                          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π c) (C …
                                        -/
  wι := Cofork.IsColimit.hom_ext hc (by simp [hg])
                                        /-
                                          🎉 no goals
                                        -/
                                                                   /-
                                                                     C : Type u_1
                                                                     inst✝¹ : CategoryTheory.Category.{?u.25155, u_1} C
                                                                     inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                                                     S S₁ S₂ S₃ : CategoryTheory.ShortComplex C
                                                                     h : S.RightHomologyData
                                                                     A : C
                                                                     hg : Eq S.g 0
                                                                     c : CategoryTheory.Limits.CokernelCofork S.f
                                                                     hc : CategoryTheory.Limits.IsColimit c
                                                                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π c) (( …
                                                                   -/
  hι := KernelFork.IsLimit.ofId _ (Cofork.IsColimit.hom_ext hc (by simp [hg]))
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


@[simp] lemma ofIsColimitCokernelCofork_g' (hg : S.g = 0) (c : CokernelCofork S.f)
  (hc : IsColimit c) : (ofIsColimitCokernelCofork S hg c hc).g' = 0 := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    hg : Eq S.g 0
    c : CategoryTheory.Limits.CokernelCofork S.f
    hc : CategoryTheory.Limits.IsColimit c
    ⊢ Eq (CategoryTheory.ShortComplex.RightHomologyData.ofIsColimitCokernelCofork  …
  -/
  rw [← cancel_epi (ofIsColimitCokernelCofork S hg c hc).p, p_g', hg, comp_zero]
  /-
    🎉 no goals
  -/


/-- When the second map `S.g` is zero, this is the right homology data on `S` given
by the chosen `cokernel S.f` -/
@[simp]
noncomputable def ofHasCokernel [HasCokernel S.f] (hg : S.g = 0) : S.RightHomologyData :=
ofIsColimitCokernelCofork S hg _ (cokernelIsCokernel _)


/-- When both `S.f` and `S.g` are zero, the middle object `S.X₂`
gives a right homology data on S -/
@[simps]
def ofZeros (hf : S.f = 0) (hg : S.g = 0) : S.RightHomologyData where
  Q := S.X₂
  H := S.X₂
  p := 𝟙 _
  ι := 𝟙 _
           /-
             C : Type u_1
             inst✝¹ : CategoryTheory.Category.{?u.30842, u_1} C
             inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
             S S₁ S₂ S₃ : CategoryTheory.ShortComplex C
             h : S.RightHomologyData
             A : C
             hf : Eq S.f 0
             hg : Eq S.g 0
             ⊢ Eq (CategoryTheory.CategoryStruct.comp S.f (CategoryTheory.CategoryStruct.id …
           -/
  wp := by rw [comp_id, hf]
           /-
             🎉 no goals
           -/
  hp := CokernelCofork.IsColimit.ofId _ hf
  wι := by
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.30842, u_1} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      S S₁ S₂ S₃ : CategoryTheory.ShortComplex C
      h : S.RightHomologyData
      A : C
      hf : Eq S.f 0
      hg : Eq S.g 0
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id S.X …
    -/
    change 𝟙 _ ≫ S.g = 0
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.30842, u_1} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      S S₁ S₂ S₃ : CategoryTheory.ShortComplex C
      h : S.RightHomologyData
      A : C
      hf : Eq S.f 0
      hg : Eq S.g 0
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id S.X …
    -/
    simp only [hg, comp_zero]
    /-
      🎉 no goals
    -/
  hι := KernelFork.IsLimit.ofId _ hg


@[simp]
lemma ofZeros_g' (hf : S.f = 0) (hg : S.g = 0) :
    (ofZeros S hf hg).g' = 0 := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    hf : Eq S.f 0
    hg : Eq S.g 0
    ⊢ Eq (CategoryTheory.ShortComplex.RightHomologyData.ofZeros S hf hg).g' 0
  -/
  rw [← cancel_epi ((ofZeros S hf hg).p), comp_zero, p_g', hg]
  /-
    🎉 no goals
  -/


/-- A short complex `S` has right homology when there exists a `S.RightHomologyData` -/
class HasRightHomology : Prop where
  condition : Nonempty S.RightHomologyData


/-- A chosen `S.RightHomologyData` for a short complex `S` that has right homology -/
noncomputable def rightHomologyData [HasRightHomology S] :
  S.RightHomologyData := HasRightHomology.condition.some


lemma mk' (h : S.RightHomologyData) : HasRightHomology S := ⟨Nonempty.intro h⟩


instance of_hasCokernel_of_hasKernel
    [HasCokernel S.f] [HasKernel (cokernel.desc S.f S.g S.zero)] :
  S.HasRightHomology := HasRightHomology.mk' (RightHomologyData.ofHasCokernelOfHasKernel S)


instance of_hasKernel {Y Z : C} (g : Y ⟶ Z) (X : C) [HasKernel g] :
    (ShortComplex.mk (0 : X ⟶ Y) g zero_comp).HasRightHomology :=
  HasRightHomology.mk' (RightHomologyData.ofHasKernel _ rfl)


instance of_hasCokernel {X Y : C} (f : X ⟶ Y) (Z : C) [HasCokernel f] :
    (ShortComplex.mk f (0 : Y ⟶ Z) comp_zero).HasRightHomology :=
  HasRightHomology.mk' (RightHomologyData.ofHasCokernel _ rfl)


instance of_zeros (X Y Z : C) :
    (ShortComplex.mk (0 : X ⟶ Y) (0 : Y ⟶ Z) zero_comp).HasRightHomology :=
  HasRightHomology.mk' (RightHomologyData.ofZeros _ rfl rfl)


/-- A right homology data for a short complex `S` induces a left homology data for `S.op`. -/
@[simps]
def op (h : S.RightHomologyData) : S.op.LeftHomologyData where
  K := Opposite.op h.Q
  H := Opposite.op h.H
  i := h.p.op
  π := h.ι.op
  wi := Quiver.Hom.unop_inj h.wp
  hi := CokernelCofork.IsColimit.ofπOp _ _ h.hp
  wπ := Quiver.Hom.unop_inj h.wι
  hπ := KernelFork.IsLimit.ofιOp _ _ h.hι


@[simp] lemma op_f' (h : S.RightHomologyData) :
    h.op.f' = h.g'.op := rfl


/-- A right homology data for a short complex `S` in the opposite category
induces a left homology data for `S.unop`. -/
@[simps]
def unop {S : ShortComplex Cᵒᵖ} (h : S.RightHomologyData) : S.unop.LeftHomologyData where
  K := Opposite.unop h.Q
  H := Opposite.unop h.H
  i := h.p.unop
  π := h.ι.unop
  wi := Quiver.Hom.op_inj h.wp
  hi := CokernelCofork.IsColimit.ofπUnop _ _ h.hp
  wπ := Quiver.Hom.op_inj h.wι
  hπ := KernelFork.IsLimit.ofιUnop _ _ h.hι


@[simp] lemma unop_f' {S : ShortComplex Cᵒᵖ} (h : S.RightHomologyData) :
    h.unop.f' = h.g'.unop := rfl


/-- A left homology data for a short complex `S` induces a right homology data for `S.op`. -/
@[simps]
def op (h : S.LeftHomologyData) : S.op.RightHomologyData where
  Q := Opposite.op h.K
  H := Opposite.op h.H
  p := h.i.op
  ι := h.π.op
  wp := Quiver.Hom.unop_inj h.wi
  hp := KernelFork.IsLimit.ofιOp _ _ h.hi
  wι := Quiver.Hom.unop_inj h.wπ
  hι := CokernelCofork.IsColimit.ofπOp _ _ h.hπ


@[simp] lemma op_g' (h : S.LeftHomologyData) :
    h.op.g' = h.f'.op := rfl


/-- A left homology data for a short complex `S` in the opposite category
induces a right homology data for `S.unop`. -/
@[simps]
def unop {S : ShortComplex Cᵒᵖ} (h : S.LeftHomologyData) : S.unop.RightHomologyData where
  Q := Opposite.unop h.K
  H := Opposite.unop h.H
  p := h.i.unop
  ι := h.π.unop
  wp := Quiver.Hom.op_inj h.wi
  hp := KernelFork.IsLimit.ofιUnop _ _ h.hi
  wι := Quiver.Hom.op_inj h.wπ
  hι := CokernelCofork.IsColimit.ofπUnop _ _ h.hπ


@[simp] lemma unop_g' {S : ShortComplex Cᵒᵖ} (h : S.LeftHomologyData) :
    h.unop.g' = h.f'.unop := rfl


instance [S.HasLeftHomology] : HasRightHomology S.op :=
  HasRightHomology.mk' S.leftHomologyData.op


instance [S.HasRightHomology] : HasLeftHomology S.op :=
  HasLeftHomology.mk' S.rightHomologyData.op


lemma hasLeftHomology_iff_op (S : ShortComplex C) :
    S.HasLeftHomology ↔ S.op.HasRightHomology :=
  ⟨fun _ => inferInstance, fun _ => HasLeftHomology.mk' S.op.rightHomologyData.unop⟩


lemma hasRightHomology_iff_op (S : ShortComplex C) :
    S.HasRightHomology ↔ S.op.HasLeftHomology :=
  ⟨fun _ => inferInstance, fun _ => HasRightHomology.mk' S.op.leftHomologyData.unop⟩


lemma hasLeftHomology_iff_unop (S : ShortComplex Cᵒᵖ) :
    S.HasLeftHomology ↔ S.unop.HasRightHomology :=
  S.unop.hasRightHomology_iff_op.symm


lemma hasRightHomology_iff_unop (S : ShortComplex Cᵒᵖ) :
    S.HasRightHomology ↔ S.unop.HasLeftHomology :=
  S.unop.hasLeftHomology_iff_op.symm


/-- Given right homology data `h₁` and `h₂` for two short complexes `S₁` and `S₂`,
a `RightHomologyMapData` for a morphism `φ : S₁ ⟶ S₂`
consists of a description of the induced morphisms on the `Q` (opcycles)
and `H` (right homology) fields of `h₁` and `h₂`. -/
structure RightHomologyMapData where
  /-- the induced map on opcycles -/
  φQ : h₁.Q ⟶ h₂.Q
  /-- the induced map on right homology -/
  φH : h₁.H ⟶ h₂.H
  /-- commutation with `p` -/
  commp : h₁.p ≫ φQ = φ.τ₂ ≫ h₂.p := by aesop_cat
  /-- commutation with `g'` -/
  commg' : φQ ≫ h₂.g' = h₁.g' ≫ φ.τ₃ := by aesop_cat
  /-- commutation with `ι` -/
  commι : φH ≫ h₂.ι = h₁.ι ≫ φQ := by aesop_cat


attribute [reassoc (attr := simp)] commp commg' commι


/-- The right homology map data associated to the zero morphism between two short complexes. -/
@[simps]
def zero (h₁ : S₁.RightHomologyData) (h₂ : S₂.RightHomologyData) :
    RightHomologyMapData 0 h₁ h₂ where
  φQ := 0
  φH := 0


/-- The right homology map data associated to the identity morphism of a short complex. -/
@[simps]
def id (h : S.RightHomologyData) : RightHomologyMapData (𝟙 S) h h where
  φQ := 𝟙 _
  φH := 𝟙 _


/-- The composition of right homology map data. -/
@[simps]
def comp {φ : S₁ ⟶ S₂} {φ' : S₂ ⟶ S₃} {h₁ : S₁.RightHomologyData}
    {h₂ : S₂.RightHomologyData} {h₃ : S₃.RightHomologyData}
    (ψ : RightHomologyMapData φ h₁ h₂) (ψ' : RightHomologyMapData φ' h₂ h₃) :
    RightHomologyMapData (φ ≫ φ') h₁ h₃ where
  φQ := ψ.φQ ≫ ψ'.φQ
  φH := ψ.φH ≫ ψ'.φH


instance : Subsingleton (RightHomologyMapData φ h₁ h₂) :=
  ⟨fun ψ₁ ψ₂ => by
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      S S₁ S₂ S₃ : CategoryTheory.ShortComplex C
      φ : Quiver.Hom S₁ S₂
      h₁ : S₁.RightHomologyData
      h₂ : S₂.RightHomologyData
      ψ₁ ψ₂ : CategoryTheory.ShortComplex.RightHomologyMapData φ h₁ h₂
      ⊢ Eq ψ₁ ψ₂
    -/
    have hQ : ψ₁.φQ = ψ₂.φQ := by rw [← cancel_epi h₁.p, commp, commp]
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      S S₁ S₂ S₃ : CategoryTheory.ShortComplex C
      φ : Quiver.Hom S₁ S₂
      h₁ : S₁.RightHomologyData
      h₂ : S₂.RightHomologyData
      ψ₁ ψ₂ : CategoryTheory.ShortComplex.RightHomologyMapData φ h₁ h₂
      hQ : Eq ψ₁.φQ ψ₂.φQ
      ⊢ Eq ψ₁ ψ₂
    -/
    have hH : ψ₁.φH = ψ₂.φH := by rw [← cancel_mono h₂.ι, commι, commι, hQ]
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      S S₁ S₂ S₃ : CategoryTheory.ShortComplex C
      φ : Quiver.Hom S₁ S₂
      h₁ : S₁.RightHomologyData
      h₂ : S₂.RightHomologyData
      ψ₁ ψ₂ : CategoryTheory.ShortComplex.RightHomologyMapData φ h₁ h₂
      hQ : Eq ψ₁.φQ ψ₂.φQ
      hH : Eq ψ₁.φH ψ₂.φH
      ⊢ Eq ψ₁ ψ₂
    -/
    cases ψ₁
    /-
      case mk
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      S S₁ S₂ S₃ : CategoryTheory.ShortComplex C
      φ : Quiver.Hom S₁ S₂
      h₁ : S₁.RightHomologyData
      h₂ : S₂.RightHomologyData
      ψ₂ : CategoryTheory.ShortComplex.RightHomologyMapData φ h₁ h₂
      φQ✝ : Quiver.Hom h₁.Q h₂.Q
      φH✝ : Quiver.Hom h₁.H h₂.H
      commp✝ : Eq (CategoryTheory.CategoryStruct.comp h₁.p φQ✝) (CategoryTheory.Cate …
      commg'✝ : Eq (CategoryTheory.CategoryStruct.comp φQ✝ h₂.g') (CategoryTheory.Ca …
      commι✝ : Eq (CategoryTheory.CategoryStruct.comp φH✝ h₂.ι) (CategoryTheory.Cate …
      hQ : Eq { φQ := φQ✝, φH := φH✝, commp := commp✝, commg' := commg'✝, commι := c …
      hH : Eq { φQ := φQ✝, φH := φH✝, commp := commp✝, commg' := commg'✝, commι := c …
      ⊢ Eq { φQ := φQ✝, φH := φH✝, commp := commp✝, commg' := commg'✝, commι := comm …
    -/
    cases ψ₂
    /-
      case mk.mk
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      S S₁ S₂ S₃ : CategoryTheory.ShortComplex C
      φ : Quiver.Hom S₁ S₂
      h₁ : S₁.RightHomologyData
      h₂ : S₂.RightHomologyData
      φQ✝¹ : Quiver.Hom h₁.Q h₂.Q
      φH✝¹ : Quiver.Hom h₁.H h₂.H
      commp✝¹ : Eq (CategoryTheory.CategoryStruct.comp h₁.p φQ✝¹) (CategoryTheory.Ca …
      commg'✝¹ : Eq (CategoryTheory.CategoryStruct.comp φQ✝¹ h₂.g') (CategoryTheory. …
      commι✝¹ : Eq (CategoryTheory.CategoryStruct.comp φH✝¹ h₂.ι) (CategoryTheory.Ca …
      φQ✝ : Quiver.Hom h₁.Q h₂.Q
      φH✝ : Quiver.Hom h₁.H h₂.H
      commp✝ : Eq (CategoryTheory.CategoryStruct.comp h₁.p φQ✝) (CategoryTheory.Cate …
      commg'✝ : Eq (CategoryTheory.CategoryStruct.comp φQ✝ h₂.g') (CategoryTheory.Ca …
      commι✝ : Eq (CategoryTheory.CategoryStruct.comp φH✝ h₂.ι) (CategoryTheory.Cate …
      hQ : Eq { φQ := φQ✝¹, φH := φH✝¹, commp := commp✝¹, commg' := commg'✝¹, commι  …
      hH : Eq { φQ := φQ✝¹, φH := φH✝¹, commp := commp✝¹, commg' := commg'✝¹, commι  …
      ⊢ Eq { φQ := φQ✝¹, φH := φH✝¹, commp := commp✝¹, commg' := commg'✝¹, commι :=  …
    -/
    congr⟩
    /-
      🎉 no goals
    -/


instance : Inhabited (RightHomologyMapData φ h₁ h₂) := ⟨by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{?u.55699, u_1} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    S S₁ S₂ S₃ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    h₁ : S₁.RightHomologyData
    h₂ : S₂.RightHomologyData
    ⊢ CategoryTheory.ShortComplex.RightHomologyMapData φ h₁ h₂
  -/
  let φQ : h₁.Q ⟶ h₂.Q := h₁.descQ (φ.τ₂ ≫ h₂.p) (by rw [← φ.comm₁₂_assoc, h₂.wp, comp_zero])
  have commg' : φQ ≫ h₂.g' = h₁.g' ≫ φ.τ₃ := by
    rw [← cancel_epi h₁.p, RightHomologyData.p_descQ_assoc, assoc,
      RightHomologyData.p_g', φ.comm₂₃, RightHomologyData.p_g'_assoc]
  let φH : h₁.H ⟶ h₂.H := h₂.liftH (h₁.ι ≫ φQ)
    (by rw [assoc, commg', RightHomologyData.ι_g'_assoc, zero_comp])
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{?u.55699, u_1} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    S S₁ S₂ S₃ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    h₁ : S₁.RightHomologyData
    h₂ : S₂.RightHomologyData
    φQ : Quiver.Hom h₁.Q h₂.Q := h₁.descQ (CategoryTheory.CategoryStruct.comp φ.τ₂ …
    commg' : Eq (CategoryTheory.CategoryStruct.comp φQ h₂.g') (CategoryTheory.Cate …
    φH : Quiver.Hom h₁.H h₂.H := h₂.liftH (CategoryTheory.CategoryStruct.comp h₁.ι …
    ⊢ CategoryTheory.ShortComplex.RightHomologyMapData φ h₁ h₂
  -/
  exact ⟨φQ, φH, by simp [φQ], commg', by simp [φH]⟩⟩
  /-
    🎉 no goals
  -/


instance : Unique (RightHomologyMapData φ h₁ h₂) := Unique.mk' _


                                                                                           /-
                                                                                             C : Type u_1
                                                                                             inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
                                                                                             inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                                                                             S₁ S₂ : CategoryTheory.ShortComplex C
                                                                                             φ : Quiver.Hom S₁ S₂
                                                                                             h₁ : S₁.RightHomologyData
                                                                                             h₂ : S₂.RightHomologyData
                                                                                             γ₁ γ₂ : CategoryTheory.ShortComplex.RightHomologyMapData φ h₁ h₂
                                                                                             eq : Eq γ₁ γ₂
                                                                                             ⊢ Eq γ₁.φH γ₂.φH
                                                                                           -/
lemma congr_φH {γ₁ γ₂ : RightHomologyMapData φ h₁ h₂} (eq : γ₁ = γ₂) : γ₁.φH = γ₂.φH := by rw [eq]
                                                                                           /-
                                                                                             🎉 no goals
                                                                                           -/

                                                                                           /-
                                                                                             C : Type u_1
                                                                                             inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
                                                                                             inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                                                                             S₁ S₂ : CategoryTheory.ShortComplex C
                                                                                             φ : Quiver.Hom S₁ S₂
                                                                                             h₁ : S₁.RightHomologyData
                                                                                             h₂ : S₂.RightHomologyData
                                                                                             γ₁ γ₂ : CategoryTheory.ShortComplex.RightHomologyMapData φ h₁ h₂
                                                                                             eq : Eq γ₁ γ₂
                                                                                             ⊢ Eq γ₁.φQ γ₂.φQ
                                                                                           -/
lemma congr_φQ {γ₁ γ₂ : RightHomologyMapData φ h₁ h₂} (eq : γ₁ = γ₂) : γ₁.φQ = γ₂.φQ := by rw [eq]
                                                                                           /-
                                                                                             🎉 no goals
                                                                                           -/


/-- When `S₁.f`, `S₁.g`, `S₂.f` and `S₂.g` are all zero, the action on right homology of a
morphism `φ : S₁ ⟶ S₂` is given by the action `φ.τ₂` on the middle objects. -/
@[simps]
def ofZeros (φ : S₁ ⟶ S₂) (hf₁ : S₁.f = 0) (hg₁ : S₁.g = 0) (hf₂ : S₂.f = 0) (hg₂ : S₂.g = 0) :
    RightHomologyMapData φ (RightHomologyData.ofZeros S₁ hf₁ hg₁)
    (RightHomologyData.ofZeros S₂ hf₂ hg₂) where
  φQ := φ.τ₂
  φH := φ.τ₂


/-- When `S₁.f` and `S₂.f` are zero and we have chosen limit kernel forks `c₁` and `c₂`
for `S₁.g` and `S₂.g` respectively, the action on right homology of a morphism `φ : S₁ ⟶ S₂` of
short complexes is given by the unique morphism `f : c₁.pt ⟶ c₂.pt` such that
`c₁.ι ≫ φ.τ₂ = f ≫ c₂.ι`. -/
@[simps]
def ofIsLimitKernelFork (φ : S₁ ⟶ S₂)
    (hf₁ : S₁.f = 0) (c₁ : KernelFork S₁.g) (hc₁ : IsLimit c₁)
    (hf₂ : S₂.f = 0) (c₂ : KernelFork S₂.g) (hc₂ : IsLimit c₂) (f : c₁.pt ⟶ c₂.pt)
    (comm : c₁.ι ≫ φ.τ₂ = f ≫ c₂.ι) :
    RightHomologyMapData φ (RightHomologyData.ofIsLimitKernelFork S₁ hf₁ c₁ hc₁)
      (RightHomologyData.ofIsLimitKernelFork S₂ hf₂ c₂ hc₂) where
  φQ := φ.τ₂
  φH := f
               /-
                 C : Type u_1
                 inst✝¹ : CategoryTheory.Category.{?u.63494, u_1} C
                 inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                 S S₁ S₂ S₃ : CategoryTheory.ShortComplex C
                 φ✝ : Quiver.Hom S₁ S₂
                 h₁ : S₁.RightHomologyData
                 h₂ : S₂.RightHomologyData
                 φ : Quiver.Hom S₁ S₂
                 hf₁ : Eq S₁.f 0
                 c₁ : CategoryTheory.Limits.KernelFork S₁.g
                 hc₁ : CategoryTheory.Limits.IsLimit c₁
                 hf₂ : Eq S₂.f 0
                 c₂ : CategoryTheory.Limits.KernelFork S₂.g
                 hc₂ : CategoryTheory.Limits.IsLimit c₂
                 f : Quiver.Hom c₁.pt c₂.pt
                 comm : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Fork.ι c₁ …
                 ⊢ Eq (CategoryTheory.CategoryStruct.comp φ.τ₂ (CategoryTheory.ShortComplex.Rig …
               -/
  commg' := by simp only [RightHomologyData.ofIsLimitKernelFork_g', φ.comm₂₃]
               /-
                 🎉 no goals
               -/
  commι := comm.symm


/-- When `S₁.g` and `S₂.g` are zero and we have chosen colimit cokernel coforks `c₁` and `c₂`
for `S₁.f` and `S₂.f` respectively, the action on right homology of a morphism `φ : S₁ ⟶ S₂` of
short complexes is given by the unique morphism `f : c₁.pt ⟶ c₂.pt` such that
`φ.τ₂ ≫ c₂.π = c₁.π ≫ f`. -/
@[simps]
def ofIsColimitCokernelCofork (φ : S₁ ⟶ S₂)
    (hg₁ : S₁.g = 0) (c₁ : CokernelCofork S₁.f) (hc₁ : IsColimit c₁)
    (hg₂ : S₂.g = 0) (c₂ : CokernelCofork S₂.f) (hc₂ : IsColimit c₂) (f : c₁.pt ⟶ c₂.pt)
    (comm : φ.τ₂ ≫ c₂.π = c₁.π ≫ f) :
    RightHomologyMapData φ (RightHomologyData.ofIsColimitCokernelCofork S₁ hg₁ c₁ hc₁)
      (RightHomologyData.ofIsColimitCokernelCofork S₂ hg₂ c₂ hc₂) where
  φQ := f
  φH := f
  commp := comm.symm


/-- When both maps `S.f` and `S.g` of a short complex `S` are zero, this is the right homology map
data (for the identity of `S`) which relates the right homology data
`RightHomologyData.ofIsLimitKernelFork` and `ofZeros` . -/
@[simps]
def compatibilityOfZerosOfIsLimitKernelFork (hf : S.f = 0) (hg : S.g = 0)
    (c : KernelFork S.g) (hc : IsLimit c) :
    RightHomologyMapData (𝟙 S)
      (RightHomologyData.ofIsLimitKernelFork S hf c hc)
      (RightHomologyData.ofZeros S hf hg) where
  φQ := 𝟙 _
  φH := c.ι


/-- When both maps `S.f` and `S.g` of a short complex `S` are zero, this is the right homology map
data (for the identity of `S`) which relates the right homology data `ofZeros` and
`ofIsColimitCokernelCofork`. -/
@[simps]
def compatibilityOfZerosOfIsColimitCokernelCofork (hf : S.f = 0) (hg : S.g = 0)
    (c : CokernelCofork S.f) (hc : IsColimit c) :
    RightHomologyMapData (𝟙 S)
      (RightHomologyData.ofZeros S hf hg)
      (RightHomologyData.ofIsColimitCokernelCofork S hg c hc) where
  φQ := c.π
  φH := c.π


/-- The right homology of a short complex,
given by the `H` field of a chosen right homology data. -/
noncomputable def rightHomology : C := S.rightHomologyData.H


/-- The "opcycles" of a short complex, given by the `Q` field of a chosen right homology data.
This is the dual notion to cycles. -/
noncomputable def opcycles : C := S.rightHomologyData.Q


/-- The canonical map `S.rightHomology ⟶ S.opcycles`. -/
noncomputable def rightHomologyι : S.rightHomology ⟶ S.opcycles :=
  S.rightHomologyData.ι


/-- The projection `S.X₂ ⟶ S.opcycles`. -/
noncomputable def pOpcycles : S.X₂ ⟶ S.opcycles := S.rightHomologyData.p


/-- The canonical map `S.opcycles ⟶ X₃`. -/
noncomputable def fromOpcycles : S.opcycles ⟶ S.X₃ := S.rightHomologyData.g'


@[reassoc (attr := simp)]
lemma f_pOpcycles : S.f ≫ S.pOpcycles = 0 := S.rightHomologyData.wp


@[reassoc (attr := simp)]
lemma p_fromOpcycles : S.pOpcycles ≫ S.fromOpcycles = S.g := S.rightHomologyData.p_g'


instance : Epi S.pOpcycles := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S S₁ S₂ S₃ : CategoryTheory.ShortComplex C
    inst✝ : S.HasRightHomology
    ⊢ CategoryTheory.Epi S.pOpcycles
  -/
  dsimp only [pOpcycles]
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S S₁ S₂ S₃ : CategoryTheory.ShortComplex C
    inst✝ : S.HasRightHomology
    ⊢ CategoryTheory.Epi S.rightHomologyData.p
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance : Mono S.rightHomologyι := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S S₁ S₂ S₃ : CategoryTheory.ShortComplex C
    inst✝ : S.HasRightHomology
    ⊢ CategoryTheory.Mono S.rightHomologyι
  -/
  dsimp only [rightHomologyι]
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S S₁ S₂ S₃ : CategoryTheory.ShortComplex C
    inst✝ : S.HasRightHomology
    ⊢ CategoryTheory.Mono S.rightHomologyData.ι
  -/
  infer_instance
  /-
    🎉 no goals
  -/


lemma rightHomology_ext_iff {A : C} (f₁ f₂ : A ⟶ S.rightHomology) :
    f₁ = f₂ ↔ f₁ ≫ S.rightHomologyι = f₂ ≫ S.rightHomologyι := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    inst✝ : S.HasRightHomology
    A : C
    f₁ f₂ : Quiver.Hom A S.rightHomology
    ⊢ Iff (Eq f₁ f₂) (Eq (CategoryTheory.CategoryStruct.comp f₁ S.rightHomologyι)  …
  -/
  rw [cancel_mono]
  /-
    🎉 no goals
  -/


@[ext]
lemma rightHomology_ext {A : C} (f₁ f₂ : A ⟶ S.rightHomology)
    (h : f₁ ≫ S.rightHomologyι = f₂ ≫ S.rightHomologyι) : f₁ = f₂ := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    inst✝ : S.HasRightHomology
    A : C
    f₁ f₂ : Quiver.Hom A S.rightHomology
    h : Eq (CategoryTheory.CategoryStruct.comp f₁ S.rightHomologyι) (CategoryTheor …
    ⊢ Eq f₁ f₂
  -/
  simpa only [rightHomology_ext_iff]
  /-
    🎉 no goals
  -/


lemma opcycles_ext_iff {A : C} (f₁ f₂ : S.opcycles ⟶ A) :
    f₁ = f₂ ↔ S.pOpcycles ≫ f₁ = S.pOpcycles ≫ f₂ := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    inst✝ : S.HasRightHomology
    A : C
    f₁ f₂ : Quiver.Hom S.opcycles A
    ⊢ Iff (Eq f₁ f₂) (Eq (CategoryTheory.CategoryStruct.comp S.pOpcycles f₁) (Cate …
  -/
  rw [cancel_epi]
  /-
    🎉 no goals
  -/


@[ext]
lemma opcycles_ext {A : C} (f₁ f₂ : S.opcycles ⟶ A)
    (h : S.pOpcycles ≫ f₁ = S.pOpcycles ≫ f₂) : f₁ = f₂ := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    inst✝ : S.HasRightHomology
    A : C
    f₁ f₂ : Quiver.Hom S.opcycles A
    h : Eq (CategoryTheory.CategoryStruct.comp S.pOpcycles f₁) (CategoryTheory.Cat …
    ⊢ Eq f₁ f₂
  -/
  simpa only [opcycles_ext_iff]
  /-
    🎉 no goals
  -/


lemma isIso_pOpcycles (hf : S.f = 0) : IsIso S.pOpcycles :=
  RightHomologyData.isIso_p _ hf


/-- When `S.f = 0`, this is the canonical isomorphism `S.opcycles ≅ S.X₂`
induced by `S.pOpcycles`. -/
@[simps! inv]
noncomputable def opcyclesIsoX₂ (hf : S.f = 0) : S.opcycles ≅ S.X₂ := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{?u.85345, u_1} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S S₁ S₂ S₃ : CategoryTheory.ShortComplex C
    inst✝ : S.HasRightHomology
    hf : Eq S.f 0
    ⊢ CategoryTheory.Iso S.opcycles S.X₂
  -/
  have := S.isIso_pOpcycles hf
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{?u.85345, u_1} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S S₁ S₂ S₃ : CategoryTheory.ShortComplex C
    inst✝ : S.HasRightHomology
    hf : Eq S.f 0
    this : CategoryTheory.IsIso S.pOpcycles
    ⊢ CategoryTheory.Iso S.opcycles S.X₂
  -/
  exact (asIso S.pOpcycles).symm
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma opcyclesIsoX₂_inv_hom_id (hf : S.f = 0) :
    S.pOpcycles ≫ (S.opcyclesIsoX₂ hf).hom = 𝟙 _ := (S.opcyclesIsoX₂ hf).inv_hom_id


@[reassoc (attr := simp)]
lemma opcyclesIsoX₂_hom_inv_id (hf : S.f = 0) :
    (S.opcyclesIsoX₂ hf).hom ≫ S.pOpcycles = 𝟙 _ := (S.opcyclesIsoX₂ hf).hom_inv_id


lemma isIso_rightHomologyι (hg : S.g = 0) : IsIso S.rightHomologyι :=
  RightHomologyData.isIso_ι _ hg


/-- When `S.g = 0`, this is the canonical isomorphism `S.opcycles ≅ S.rightHomology` induced
by `S.rightHomologyι`. -/
@[simps! inv]
noncomputable def opcyclesIsoRightHomology (hg : S.g = 0) : S.opcycles ≅ S.rightHomology := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{?u.88223, u_1} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S S₁ S₂ S₃ : CategoryTheory.ShortComplex C
    inst✝ : S.HasRightHomology
    hg : Eq S.g 0
    ⊢ CategoryTheory.Iso S.opcycles S.rightHomology
  -/
  have := S.isIso_rightHomologyι hg
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{?u.88223, u_1} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S S₁ S₂ S₃ : CategoryTheory.ShortComplex C
    inst✝ : S.HasRightHomology
    hg : Eq S.g 0
    this : CategoryTheory.IsIso S.rightHomologyι
    ⊢ CategoryTheory.Iso S.opcycles S.rightHomology
  -/
  exact (asIso S.rightHomologyι).symm
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma opcyclesIsoRightHomology_inv_hom_id (hg : S.g = 0) :
    S.rightHomologyι ≫ (S.opcyclesIsoRightHomology hg).hom = 𝟙 _ :=
  (S.opcyclesIsoRightHomology hg).inv_hom_id


@[reassoc (attr := simp)]
lemma opcyclesIsoRightHomology_hom_inv_id (hg : S.g = 0) :
    (S.opcyclesIsoRightHomology hg).hom ≫ S.rightHomologyι  = 𝟙 _ :=
  (S.opcyclesIsoRightHomology hg).hom_inv_id


/-- The (unique) right homology map data associated to a morphism of short complexes that
are both equipped with right homology data. -/
def rightHomologyMapData : RightHomologyMapData φ h₁ h₂ := default


/-- Given a morphism `φ : S₁ ⟶ S₂` of short complexes and right homology data `h₁` and `h₂`
for `S₁` and `S₂` respectively, this is the induced right homology map `h₁.H ⟶ h₁.H`. -/
def rightHomologyMap' : h₁.H ⟶ h₂.H := (rightHomologyMapData φ _ _).φH


/-- Given a morphism `φ : S₁ ⟶ S₂` of short complexes and right homology data `h₁` and `h₂`
for `S₁` and `S₂` respectively, this is the induced morphism `h₁.K ⟶ h₁.K` on opcycles. -/
def opcyclesMap' : h₁.Q ⟶ h₂.Q := (rightHomologyMapData φ _ _).φQ


@[reassoc (attr := simp)]
lemma p_opcyclesMap' : h₁.p ≫ opcyclesMap' φ h₁ h₂ = φ.τ₂ ≫ h₂.p :=
  RightHomologyMapData.commp _


@[reassoc (attr := simp)]
lemma opcyclesMap'_g' : opcyclesMap' φ h₁ h₂ ≫ h₂.g' = h₁.g' ≫ φ.τ₃ := by
  simp only [← cancel_epi h₁.p, assoc, φ.comm₂₃, p_opcyclesMap'_assoc,
    RightHomologyData.p_g'_assoc, RightHomologyData.p_g']


@[reassoc (attr := simp)]
lemma rightHomologyι_naturality' :
    rightHomologyMap' φ h₁ h₂ ≫ h₂.ι = h₁.ι ≫ opcyclesMap' φ h₁ h₂ :=
  RightHomologyMapData.commι _


/-- The (right) homology map `S₁.rightHomology ⟶ S₂.rightHomology` induced by a morphism
`S₁ ⟶ S₂` of short complexes. -/
noncomputable def rightHomologyMap : S₁.rightHomology ⟶ S₂.rightHomology :=
  rightHomologyMap' φ _ _


/-- The morphism `S₁.opcycles ⟶ S₂.opcycles` induced by a morphism `S₁ ⟶ S₂` of short complexes. -/
noncomputable def opcyclesMap : S₁.opcycles ⟶ S₂.opcycles :=
  opcyclesMap' φ _ _


@[reassoc (attr := simp)]
lemma p_opcyclesMap : S₁.pOpcycles ≫ opcyclesMap φ = φ.τ₂ ≫ S₂.pOpcycles :=
  p_opcyclesMap' _ _ _


@[reassoc (attr := simp)]
lemma fromOpcycles_naturality : opcyclesMap φ ≫ S₂.fromOpcycles = S₁.fromOpcycles ≫ φ.τ₃ :=
  opcyclesMap'_g' _ _ _


@[reassoc (attr := simp)]
lemma rightHomologyι_naturality :
    rightHomologyMap φ ≫ S₂.rightHomologyι = S₁.rightHomologyι ≫ opcyclesMap φ :=
  rightHomologyι_naturality' _ _ _


lemma rightHomologyMap'_eq : rightHomologyMap' φ h₁ h₂ = γ.φH :=
  RightHomologyMapData.congr_φH (Subsingleton.elim _ _)


lemma opcyclesMap'_eq : opcyclesMap' φ h₁ h₂ = γ.φQ :=
  RightHomologyMapData.congr_φQ (Subsingleton.elim _ _)


@[simp]
lemma rightHomologyMap'_id (h : S.RightHomologyData) :
    rightHomologyMap' (𝟙 S) h h = 𝟙 _ :=
  (RightHomologyMapData.id h).rightHomologyMap'_eq


@[simp]
lemma opcyclesMap'_id (h : S.RightHomologyData) :
    opcyclesMap' (𝟙 S) h h = 𝟙 _ :=
  (RightHomologyMapData.id h).opcyclesMap'_eq


@[simp]
lemma rightHomologyMap_id [HasRightHomology S] :
    rightHomologyMap (𝟙 S) = 𝟙 _ :=
  rightHomologyMap'_id _


@[simp]
lemma opcyclesMap_id [HasRightHomology S] :
    opcyclesMap (𝟙 S) = 𝟙 _ :=
  opcyclesMap'_id _


@[simp]
lemma rightHomologyMap'_zero (h₁ : S₁.RightHomologyData) (h₂ : S₂.RightHomologyData) :
    rightHomologyMap' 0 h₁ h₂ = 0 :=
  (RightHomologyMapData.zero h₁ h₂).rightHomologyMap'_eq


@[simp]
lemma opcyclesMap'_zero (h₁ : S₁.RightHomologyData) (h₂ : S₂.RightHomologyData) :
    opcyclesMap' 0 h₁ h₂ = 0 :=
  (RightHomologyMapData.zero h₁ h₂).opcyclesMap'_eq


@[simp]
lemma rightHomologyMap_zero [HasRightHomology S₁] [HasRightHomology S₂] :
    rightHomologyMap (0 : S₁ ⟶ S₂) = 0 :=
  rightHomologyMap'_zero _ _


@[simp]
lemma opcyclesMap_zero [HasRightHomology S₁] [HasRightHomology S₂] :
    opcyclesMap (0 : S₁ ⟶ S₂) = 0 :=
  opcyclesMap'_zero _ _


@[reassoc]
lemma rightHomologyMap'_comp (φ₁ : S₁ ⟶ S₂) (φ₂ : S₂ ⟶ S₃)
    (h₁ : S₁.RightHomologyData) (h₂ : S₂.RightHomologyData) (h₃ : S₃.RightHomologyData) :
    rightHomologyMap' (φ₁ ≫ φ₂) h₁ h₃ = rightHomologyMap' φ₁ h₁ h₂ ≫
      rightHomologyMap' φ₂ h₂ h₃ := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ S₃ : CategoryTheory.ShortComplex C
    φ₁ : Quiver.Hom S₁ S₂
    φ₂ : Quiver.Hom S₂ S₃
    h₁ : S₁.RightHomologyData
    h₂ : S₂.RightHomologyData
    h₃ : S₃.RightHomologyData
    ⊢ Eq (CategoryTheory.ShortComplex.rightHomologyMap' (CategoryTheory.CategorySt …
  -/
  let γ₁ := rightHomologyMapData φ₁ h₁ h₂
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ S₃ : CategoryTheory.ShortComplex C
    φ₁ : Quiver.Hom S₁ S₂
    φ₂ : Quiver.Hom S₂ S₃
    h₁ : S₁.RightHomologyData
    h₂ : S₂.RightHomologyData
    h₃ : S₃.RightHomologyData
    γ₁ : CategoryTheory.ShortComplex.RightHomologyMapData φ₁ h₁ h₂ := CategoryTheo …
    ⊢ Eq (CategoryTheory.ShortComplex.rightHomologyMap' (CategoryTheory.CategorySt …
  -/
  let γ₂ := rightHomologyMapData φ₂ h₂ h₃
  rw [γ₁.rightHomologyMap'_eq, γ₂.rightHomologyMap'_eq, (γ₁.comp γ₂).rightHomologyMap'_eq,
    RightHomologyMapData.comp_φH]


@[reassoc]
lemma opcyclesMap'_comp (φ₁ : S₁ ⟶ S₂) (φ₂ : S₂ ⟶ S₃)
    (h₁ : S₁.RightHomologyData) (h₂ : S₂.RightHomologyData) (h₃ : S₃.RightHomologyData) :
    opcyclesMap' (φ₁ ≫ φ₂) h₁ h₃ = opcyclesMap' φ₁ h₁ h₂ ≫ opcyclesMap' φ₂ h₂ h₃ := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ S₃ : CategoryTheory.ShortComplex C
    φ₁ : Quiver.Hom S₁ S₂
    φ₂ : Quiver.Hom S₂ S₃
    h₁ : S₁.RightHomologyData
    h₂ : S₂.RightHomologyData
    h₃ : S₃.RightHomologyData
    ⊢ Eq (CategoryTheory.ShortComplex.opcyclesMap' (CategoryTheory.CategoryStruct. …
  -/
  let γ₁ := rightHomologyMapData φ₁ h₁ h₂
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ S₃ : CategoryTheory.ShortComplex C
    φ₁ : Quiver.Hom S₁ S₂
    φ₂ : Quiver.Hom S₂ S₃
    h₁ : S₁.RightHomologyData
    h₂ : S₂.RightHomologyData
    h₃ : S₃.RightHomologyData
    γ₁ : CategoryTheory.ShortComplex.RightHomologyMapData φ₁ h₁ h₂ := CategoryTheo …
    ⊢ Eq (CategoryTheory.ShortComplex.opcyclesMap' (CategoryTheory.CategoryStruct. …
  -/
  let γ₂ := rightHomologyMapData φ₂ h₂ h₃
  rw [γ₁.opcyclesMap'_eq, γ₂.opcyclesMap'_eq, (γ₁.comp γ₂).opcyclesMap'_eq,
    RightHomologyMapData.comp_φQ]


@[simp]
lemma rightHomologyMap_comp [HasRightHomology S₁] [HasRightHomology S₂] [HasRightHomology S₃]
    (φ₁ : S₁ ⟶ S₂) (φ₂ : S₂ ⟶ S₃) :
    rightHomologyMap (φ₁ ≫ φ₂) = rightHomologyMap φ₁ ≫ rightHomologyMap φ₂ :=
rightHomologyMap'_comp _ _ _ _ _


@[simp]
lemma opcyclesMap_comp [HasRightHomology S₁] [HasRightHomology S₂] [HasRightHomology S₃]
    (φ₁ : S₁ ⟶ S₂) (φ₂ : S₂ ⟶ S₃) :
    opcyclesMap (φ₁ ≫ φ₂) = opcyclesMap φ₁ ≫ opcyclesMap φ₂ :=
  opcyclesMap'_comp _ _ _ _ _


/-- An isomorphism of short complexes `S₁ ≅ S₂` induces an isomorphism on the `H` fields
of right homology data of `S₁` and `S₂`. -/
@[simps]
def rightHomologyMapIso' (e : S₁ ≅ S₂) (h₁ : S₁.RightHomologyData)
    (h₂ : S₂.RightHomologyData) : h₁.H ≅ h₂.H where
  hom := rightHomologyMap' e.hom h₁ h₂
  inv := rightHomologyMap' e.inv h₂ h₁
                   /-
                     C : Type u_1
                     inst✝¹ : CategoryTheory.Category.{?u.115410, u_1} C
                     inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                     S S₁ S₂ S₃ : CategoryTheory.ShortComplex C
                     e : CategoryTheory.Iso S₁ S₂
                     h₁ : S₁.RightHomologyData
                     h₂ : S₂.RightHomologyData
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.ShortComplex.rightHom …
                   -/
  hom_inv_id := by rw [← rightHomologyMap'_comp, e.hom_inv_id, rightHomologyMap'_id]
                   /-
                     🎉 no goals
                   -/
                   /-
                     C : Type u_1
                     inst✝¹ : CategoryTheory.Category.{?u.115410, u_1} C
                     inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                     S S₁ S₂ S₃ : CategoryTheory.ShortComplex C
                     e : CategoryTheory.Iso S₁ S₂
                     h₁ : S₁.RightHomologyData
                     h₂ : S₂.RightHomologyData
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.ShortComplex.rightHom …
                   -/
  inv_hom_id := by rw [← rightHomologyMap'_comp, e.inv_hom_id, rightHomologyMap'_id]
                   /-
                     🎉 no goals
                   -/


instance isIso_rightHomologyMap'_of_isIso (φ : S₁ ⟶ S₂) [IsIso φ]
    (h₁ : S₁.RightHomologyData) (h₂ : S₂.RightHomologyData) :
    IsIso (rightHomologyMap' φ h₁ h₂) :=
  (inferInstance : IsIso (rightHomologyMapIso' (asIso φ) h₁ h₂).hom)


/-- An isomorphism of short complexes `S₁ ≅ S₂` induces an isomorphism on the `Q` fields
of right homology data of `S₁` and `S₂`. -/
@[simps]
def opcyclesMapIso' (e : S₁ ≅ S₂) (h₁ : S₁.RightHomologyData)
    (h₂ : S₂.RightHomologyData) : h₁.Q ≅ h₂.Q where
  hom := opcyclesMap' e.hom h₁ h₂
  inv := opcyclesMap' e.inv h₂ h₁
                   /-
                     C : Type u_1
                     inst✝¹ : CategoryTheory.Category.{?u.118102, u_1} C
                     inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                     S S₁ S₂ S₃ : CategoryTheory.ShortComplex C
                     e : CategoryTheory.Iso S₁ S₂
                     h₁ : S₁.RightHomologyData
                     h₂ : S₂.RightHomologyData
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.ShortComplex.opcycles …
                   -/
  hom_inv_id := by rw [← opcyclesMap'_comp, e.hom_inv_id, opcyclesMap'_id]
                   /-
                     🎉 no goals
                   -/
                   /-
                     C : Type u_1
                     inst✝¹ : CategoryTheory.Category.{?u.118102, u_1} C
                     inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                     S S₁ S₂ S₃ : CategoryTheory.ShortComplex C
                     e : CategoryTheory.Iso S₁ S₂
                     h₁ : S₁.RightHomologyData
                     h₂ : S₂.RightHomologyData
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.ShortComplex.opcycles …
                   -/
  inv_hom_id := by rw [← opcyclesMap'_comp, e.inv_hom_id, opcyclesMap'_id]
                   /-
                     🎉 no goals
                   -/


instance isIso_opcyclesMap'_of_isIso (φ : S₁ ⟶ S₂) [IsIso φ]
    (h₁ : S₁.RightHomologyData) (h₂ : S₂.RightHomologyData) :
    IsIso (opcyclesMap' φ h₁ h₂) :=
  (inferInstance : IsIso (opcyclesMapIso' (asIso φ) h₁ h₂).hom)


/-- The isomorphism `S₁.rightHomology ≅ S₂.rightHomology` induced by an isomorphism of
short complexes `S₁ ≅ S₂`. -/
@[simps]
noncomputable def rightHomologyMapIso (e : S₁ ≅ S₂) [S₁.HasRightHomology]
    [S₂.HasRightHomology] : S₁.rightHomology ≅ S₂.rightHomology where
  hom := rightHomologyMap e.hom
  inv := rightHomologyMap e.inv
                   /-
                     C : Type u_1
                     inst✝³ : CategoryTheory.Category.{?u.120786, u_1} C
                     inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
                     S S₁ S₂ S₃ : CategoryTheory.ShortComplex C
                     e : CategoryTheory.Iso S₁ S₂
                     inst✝¹ : S₁.HasRightHomology
                     inst✝ : S₂.HasRightHomology
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.ShortComplex.rightHom …
                   -/
  hom_inv_id := by rw [← rightHomologyMap_comp, e.hom_inv_id, rightHomologyMap_id]
                   /-
                     🎉 no goals
                   -/
                   /-
                     C : Type u_1
                     inst✝³ : CategoryTheory.Category.{?u.120786, u_1} C
                     inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
                     S S₁ S₂ S₃ : CategoryTheory.ShortComplex C
                     e : CategoryTheory.Iso S₁ S₂
                     inst✝¹ : S₁.HasRightHomology
                     inst✝ : S₂.HasRightHomology
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.ShortComplex.rightHom …
                   -/
  inv_hom_id := by rw [← rightHomologyMap_comp, e.inv_hom_id, rightHomologyMap_id]
                   /-
                     🎉 no goals
                   -/


instance isIso_rightHomologyMap_of_iso (φ : S₁ ⟶ S₂) [IsIso φ] [S₁.HasRightHomology]
    [S₂.HasRightHomology] :
    IsIso (rightHomologyMap φ) :=
  (inferInstance : IsIso (rightHomologyMapIso (asIso φ)).hom)


/-- The isomorphism `S₁.opcycles ≅ S₂.opcycles` induced by an isomorphism
of short complexes `S₁ ≅ S₂`. -/
@[simps]
noncomputable def opcyclesMapIso (e : S₁ ≅ S₂) [S₁.HasRightHomology]
    [S₂.HasRightHomology] : S₁.opcycles ≅ S₂.opcycles where
  hom := opcyclesMap e.hom
  inv := opcyclesMap e.inv
                   /-
                     C : Type u_1
                     inst✝³ : CategoryTheory.Category.{?u.124600, u_1} C
                     inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
                     S S₁ S₂ S₃ : CategoryTheory.ShortComplex C
                     e : CategoryTheory.Iso S₁ S₂
                     inst✝¹ : S₁.HasRightHomology
                     inst✝ : S₂.HasRightHomology
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.ShortComplex.opcycles …
                   -/
  hom_inv_id := by rw [← opcyclesMap_comp, e.hom_inv_id, opcyclesMap_id]
                   /-
                     🎉 no goals
                   -/
                   /-
                     C : Type u_1
                     inst✝³ : CategoryTheory.Category.{?u.124600, u_1} C
                     inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
                     S S₁ S₂ S₃ : CategoryTheory.ShortComplex C
                     e : CategoryTheory.Iso S₁ S₂
                     inst✝¹ : S₁.HasRightHomology
                     inst✝ : S₂.HasRightHomology
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.ShortComplex.opcycles …
                   -/
  inv_hom_id := by rw [← opcyclesMap_comp, e.inv_hom_id, opcyclesMap_id]
                   /-
                     🎉 no goals
                   -/


instance isIso_opcyclesMap_of_iso (φ : S₁ ⟶ S₂) [IsIso φ] [S₁.HasRightHomology]
    [S₂.HasRightHomology] : IsIso (opcyclesMap φ) :=
  (inferInstance : IsIso (opcyclesMapIso (asIso φ)).hom)


/-- The isomorphism `S.rightHomology ≅ h.H` induced by a right homology data `h` for a
short complex `S`. -/
noncomputable def rightHomologyIso : S.rightHomology ≅ h.H :=
  rightHomologyMapIso' (Iso.refl _) _ _


/-- The isomorphism `S.opcycles ≅ h.Q` induced by a right homology data `h` for a
short complex `S`. -/
noncomputable def opcyclesIso : S.opcycles ≅ h.Q :=
  opcyclesMapIso' (Iso.refl _) _ _


@[reassoc (attr := simp)]
lemma p_comp_opcyclesIso_inv : h.p ≫ h.opcyclesIso.inv = S.pOpcycles := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    h : S.RightHomologyData
    inst✝ : S.HasRightHomology
    ⊢ Eq (CategoryTheory.CategoryStruct.comp h.p h.opcyclesIso.inv) S.pOpcycles
  -/
  dsimp [pOpcycles, RightHomologyData.opcyclesIso]
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    h : S.RightHomologyData
    inst✝ : S.HasRightHomology
    ⊢ Eq (CategoryTheory.CategoryStruct.comp h.p (CategoryTheory.ShortComplex.opcy …
  -/
  simp only [p_opcyclesMap', id_τ₂, id_comp]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma pOpcycles_comp_opcyclesIso_hom : S.pOpcycles ≫ h.opcyclesIso.hom = h.p := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    h : S.RightHomologyData
    inst✝ : S.HasRightHomology
    ⊢ Eq (CategoryTheory.CategoryStruct.comp S.pOpcycles h.opcyclesIso.hom) h.p
  -/
  simp only [← h.p_comp_opcyclesIso_inv, assoc, Iso.inv_hom_id, comp_id]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma rightHomologyIso_inv_comp_rightHomologyι :
    h.rightHomologyIso.inv ≫ S.rightHomologyι = h.ι ≫ h.opcyclesIso.inv := by
  dsimp only [rightHomologyι, rightHomologyIso, opcyclesIso, rightHomologyMapIso',
    opcyclesMapIso', Iso.refl]
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    h : S.RightHomologyData
    inst✝ : S.HasRightHomology
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.ShortComplex.rightHom …
  -/
  rw [rightHomologyι_naturality']
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma rightHomologyIso_hom_comp_ι :
    h.rightHomologyIso.hom ≫ h.ι = S.rightHomologyι ≫ h.opcyclesIso.hom := by
  simp only [← cancel_mono h.opcyclesIso.inv, ← cancel_epi h.rightHomologyIso.inv,
    assoc, Iso.inv_hom_id_assoc, Iso.hom_inv_id, comp_id, rightHomologyIso_inv_comp_rightHomologyι]


lemma rightHomologyMap_eq [S₁.HasRightHomology] [S₂.HasRightHomology] :
    rightHomologyMap φ = h₁.rightHomologyIso.hom ≫ γ.φH ≫ h₂.rightHomologyIso.inv := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    h₁ : S₁.RightHomologyData
    h₂ : S₂.RightHomologyData
    γ : CategoryTheory.ShortComplex.RightHomologyMapData φ h₁ h₂
    inst✝¹ : S₁.HasRightHomology
    inst✝ : S₂.HasRightHomology
    ⊢ Eq (CategoryTheory.ShortComplex.rightHomologyMap φ) (CategoryTheory.Category …
  -/
  dsimp [RightHomologyData.rightHomologyIso, rightHomologyMapIso']
  rw [← γ.rightHomologyMap'_eq, ← rightHomologyMap'_comp,
    ← rightHomologyMap'_comp, id_comp, comp_id]
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    h₁ : S₁.RightHomologyData
    h₂ : S₂.RightHomologyData
    γ : CategoryTheory.ShortComplex.RightHomologyMapData φ h₁ h₂
    inst✝¹ : S₁.HasRightHomology
    inst✝ : S₂.HasRightHomology
    ⊢ Eq (CategoryTheory.ShortComplex.rightHomologyMap φ) (CategoryTheory.ShortCom …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma opcyclesMap_eq [S₁.HasRightHomology] [S₂.HasRightHomology] :
    opcyclesMap φ = h₁.opcyclesIso.hom ≫ γ.φQ ≫ h₂.opcyclesIso.inv := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    h₁ : S₁.RightHomologyData
    h₂ : S₂.RightHomologyData
    γ : CategoryTheory.ShortComplex.RightHomologyMapData φ h₁ h₂
    inst✝¹ : S₁.HasRightHomology
    inst✝ : S₂.HasRightHomology
    ⊢ Eq (CategoryTheory.ShortComplex.opcyclesMap φ) (CategoryTheory.CategoryStruc …
  -/
  dsimp [RightHomologyData.opcyclesIso, cyclesMapIso']
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    h₁ : S₁.RightHomologyData
    h₂ : S₂.RightHomologyData
    γ : CategoryTheory.ShortComplex.RightHomologyMapData φ h₁ h₂
    inst✝¹ : S₁.HasRightHomology
    inst✝ : S₂.HasRightHomology
    ⊢ Eq (CategoryTheory.ShortComplex.opcyclesMap φ) (CategoryTheory.CategoryStruc …
  -/
  rw [← γ.opcyclesMap'_eq, ← opcyclesMap'_comp, ← opcyclesMap'_comp, id_comp, comp_id]
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    h₁ : S₁.RightHomologyData
    h₂ : S₂.RightHomologyData
    γ : CategoryTheory.ShortComplex.RightHomologyMapData φ h₁ h₂
    inst✝¹ : S₁.HasRightHomology
    inst✝ : S₂.HasRightHomology
    ⊢ Eq (CategoryTheory.ShortComplex.opcyclesMap φ) (CategoryTheory.ShortComplex. …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma rightHomologyMap_comm [S₁.HasRightHomology] [S₂.HasRightHomology] :
    rightHomologyMap φ ≫ h₂.rightHomologyIso.hom = h₁.rightHomologyIso.hom ≫ γ.φH := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    h₁ : S₁.RightHomologyData
    h₂ : S₂.RightHomologyData
    γ : CategoryTheory.ShortComplex.RightHomologyMapData φ h₁ h₂
    inst✝¹ : S₁.HasRightHomology
    inst✝ : S₂.HasRightHomology
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.ShortComplex.rightHom …
  -/
  simp only [γ.rightHomologyMap_eq, assoc, Iso.inv_hom_id, comp_id]
  /-
    🎉 no goals
  -/


lemma opcyclesMap_comm [S₁.HasRightHomology] [S₂.HasRightHomology] :
    opcyclesMap φ ≫ h₂.opcyclesIso.hom = h₁.opcyclesIso.hom ≫ γ.φQ := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    h₁ : S₁.RightHomologyData
    h₂ : S₂.RightHomologyData
    γ : CategoryTheory.ShortComplex.RightHomologyMapData φ h₁ h₂
    inst✝¹ : S₁.HasRightHomology
    inst✝ : S₂.HasRightHomology
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.ShortComplex.opcycles …
  -/
  simp only [γ.opcyclesMap_eq, assoc, Iso.inv_hom_id, comp_id]
  /-
    🎉 no goals
  -/


/-- The right homology functor `ShortComplex C ⥤ C`, where the right homology of a
short complex `S` is understood as a kernel of the obvious map `S.fromOpcycles : S.opcycles ⟶ S.X₃`
where `S.opcycles` is a cokernel of `S.f : S.X₁ ⟶ S.X₂`. -/
@[simps]
noncomputable def rightHomologyFunctor : ShortComplex C ⥤ C where
  obj S := S.rightHomology
  map := rightHomologyMap


/-- The opcycles functor `ShortComplex C ⥤ C` which sends a short complex `S` to `S.opcycles`
which is a cokernel of `S.f : S.X₁ ⟶ S.X₂`. -/
@[simps]
noncomputable def opcyclesFunctor :
    ShortComplex C ⥤ C where
  obj S := S.opcycles
  map := opcyclesMap


/-- The natural transformation `S.rightHomology ⟶ S.opcycles` for all short complexes `S`. -/
@[simps]
noncomputable def rightHomologyιNatTrans :
    rightHomologyFunctor C ⟶ opcyclesFunctor C where
  app S := rightHomologyι S
  naturality := fun _ _ φ => rightHomologyι_naturality φ


/-- The natural transformation `S.X₂ ⟶ S.opcycles` for all short complexes `S`. -/
@[simps]
noncomputable def pOpcyclesNatTrans :
    ShortComplex.π₂ ⟶ opcyclesFunctor C where
  app S := S.pOpcycles


/-- The natural transformation `S.opcycles ⟶ S.X₃` for all short complexes `S`. -/
@[simps]
noncomputable def fromOpcyclesNatTrans :
    opcyclesFunctor C ⟶ π₃ where
  app S := S.fromOpcycles
  naturality := fun _ _  φ => fromOpcycles_naturality φ


/-- A left homology map data for a morphism of short complexes induces
a right homology map data in the opposite category. -/
@[simps]
def LeftHomologyMapData.op {S₁ S₂ : ShortComplex C} {φ : S₁ ⟶ S₂}
    {h₁ : S₁.LeftHomologyData} {h₂ : S₂.LeftHomologyData}
    (ψ : LeftHomologyMapData φ h₁ h₂) : RightHomologyMapData (opMap φ) h₂.op h₁.op where
  φQ := ψ.φK.op
  φH := ψ.φH.op
                                   /-
                                     C : Type u_1
                                     inst✝¹ : CategoryTheory.Category.{?u.158593, u_1} C
                                     inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                     S S₁✝ S₂✝ S₃ S₁ S₂ : CategoryTheory.ShortComplex C
                                     φ : Quiver.Hom S₁ S₂
                                     h₁ : S₁.LeftHomologyData
                                     h₂ : S₂.LeftHomologyData
                                     ψ : CategoryTheory.ShortComplex.LeftHomologyMapData φ h₁ h₂
                                     ⊢ Eq (CategoryTheory.CategoryStruct.comp h₂.op.p ψ.φK.op).unop (CategoryTheory …
                                   -/
  commp := Quiver.Hom.unop_inj (by simp)
                                   /-
                                     🎉 no goals
                                   -/
                                    /-
                                      C : Type u_1
                                      inst✝¹ : CategoryTheory.Category.{?u.158593, u_1} C
                                      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                      S S₁✝ S₂✝ S₃ S₁ S₂ : CategoryTheory.ShortComplex C
                                      φ : Quiver.Hom S₁ S₂
                                      h₁ : S₁.LeftHomologyData
                                      h₂ : S₂.LeftHomologyData
                                      ψ : CategoryTheory.ShortComplex.LeftHomologyMapData φ h₁ h₂
                                      ⊢ Eq (CategoryTheory.CategoryStruct.comp ψ.φK.op h₁.op.g').unop (CategoryTheor …
                                    -/
  commg' := Quiver.Hom.unop_inj (by simp)
                                    /-
                                      🎉 no goals
                                    -/
                                   /-
                                     C : Type u_1
                                     inst✝¹ : CategoryTheory.Category.{?u.158593, u_1} C
                                     inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                     S S₁✝ S₂✝ S₃ S₁ S₂ : CategoryTheory.ShortComplex C
                                     φ : Quiver.Hom S₁ S₂
                                     h₁ : S₁.LeftHomologyData
                                     h₂ : S₂.LeftHomologyData
                                     ψ : CategoryTheory.ShortComplex.LeftHomologyMapData φ h₁ h₂
                                     ⊢ Eq (CategoryTheory.CategoryStruct.comp ψ.φH.op h₁.op.ι).unop (CategoryTheory …
                                   -/
  commι := Quiver.Hom.unop_inj (by simp)
                                   /-
                                     🎉 no goals
                                   -/


/-- A left homology map data for a morphism of short complexes in the opposite category
induces a right homology map data in the original category. -/
@[simps]
def LeftHomologyMapData.unop {S₁ S₂ : ShortComplex Cᵒᵖ} {φ : S₁ ⟶ S₂}
    {h₁ : S₁.LeftHomologyData} {h₂ : S₂.LeftHomologyData}
    (ψ : LeftHomologyMapData φ h₁ h₂) : RightHomologyMapData (unopMap φ) h₂.unop h₁.unop where
  φQ := ψ.φK.unop
  φH := ψ.φH.unop
                                 /-
                                   C : Type u_1
                                   inst✝¹ : CategoryTheory.Category.{?u.161346, u_1} C
                                   inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                   S S₁✝ S₂✝ S₃ : CategoryTheory.ShortComplex C
                                   S₁ S₂ : CategoryTheory.ShortComplex (Opposite C)
                                   φ : Quiver.Hom S₁ S₂
                                   h₁ : S₁.LeftHomologyData
                                   h₂ : S₂.LeftHomologyData
                                   ψ : CategoryTheory.ShortComplex.LeftHomologyMapData φ h₁ h₂
                                   ⊢ Eq (CategoryTheory.CategoryStruct.comp h₂.unop.p ψ.φK.unop).op (CategoryTheo …
                                 -/
  commp := Quiver.Hom.op_inj (by simp)
                                 /-
                                   🎉 no goals
                                 -/
                                  /-
                                    C : Type u_1
                                    inst✝¹ : CategoryTheory.Category.{?u.161346, u_1} C
                                    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                    S S₁✝ S₂✝ S₃ : CategoryTheory.ShortComplex C
                                    S₁ S₂ : CategoryTheory.ShortComplex (Opposite C)
                                    φ : Quiver.Hom S₁ S₂
                                    h₁ : S₁.LeftHomologyData
                                    h₂ : S₂.LeftHomologyData
                                    ψ : CategoryTheory.ShortComplex.LeftHomologyMapData φ h₁ h₂
                                    ⊢ Eq (CategoryTheory.CategoryStruct.comp ψ.φK.unop h₁.unop.g').op (CategoryThe …
                                  -/
  commg' := Quiver.Hom.op_inj (by simp)
                                  /-
                                    🎉 no goals
                                  -/
                                 /-
                                   C : Type u_1
                                   inst✝¹ : CategoryTheory.Category.{?u.161346, u_1} C
                                   inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                   S S₁✝ S₂✝ S₃ : CategoryTheory.ShortComplex C
                                   S₁ S₂ : CategoryTheory.ShortComplex (Opposite C)
                                   φ : Quiver.Hom S₁ S₂
                                   h₁ : S₁.LeftHomologyData
                                   h₂ : S₂.LeftHomologyData
                                   ψ : CategoryTheory.ShortComplex.LeftHomologyMapData φ h₁ h₂
                                   ⊢ Eq (CategoryTheory.CategoryStruct.comp ψ.φH.unop h₁.unop.ι).op (CategoryTheo …
                                 -/
  commι := Quiver.Hom.op_inj (by simp)
                                 /-
                                   🎉 no goals
                                 -/


/-- A right homology map data for a morphism of short complexes induces
a left homology map data in the opposite category. -/
@[simps]
def RightHomologyMapData.op {S₁ S₂ : ShortComplex C} {φ : S₁ ⟶ S₂}
    {h₁ : S₁.RightHomologyData} {h₂ : S₂.RightHomologyData}
    (ψ : RightHomologyMapData φ h₁ h₂) : LeftHomologyMapData (opMap φ) h₂.op h₁.op where
  φK := ψ.φQ.op
  φH := ψ.φH.op
                                   /-
                                     C : Type u_1
                                     inst✝¹ : CategoryTheory.Category.{?u.163958, u_1} C
                                     inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                     S S₁✝ S₂✝ S₃ S₁ S₂ : CategoryTheory.ShortComplex C
                                     φ : Quiver.Hom S₁ S₂
                                     h₁ : S₁.RightHomologyData
                                     h₂ : S₂.RightHomologyData
                                     ψ : CategoryTheory.ShortComplex.RightHomologyMapData φ h₁ h₂
                                     ⊢ Eq (CategoryTheory.CategoryStruct.comp ψ.φQ.op h₁.op.i).unop (CategoryTheory …
                                   -/
  commi := Quiver.Hom.unop_inj (by simp)
                                   /-
                                     🎉 no goals
                                   -/
                                    /-
                                      C : Type u_1
                                      inst✝¹ : CategoryTheory.Category.{?u.163958, u_1} C
                                      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                      S S₁✝ S₂✝ S₃ S₁ S₂ : CategoryTheory.ShortComplex C
                                      φ : Quiver.Hom S₁ S₂
                                      h₁ : S₁.RightHomologyData
                                      h₂ : S₂.RightHomologyData
                                      ψ : CategoryTheory.ShortComplex.RightHomologyMapData φ h₁ h₂
                                      ⊢ Eq (CategoryTheory.CategoryStruct.comp h₂.op.f' ψ.φQ.op).unop (CategoryTheor …
                                    -/
  commf' := Quiver.Hom.unop_inj (by simp)
                                    /-
                                      🎉 no goals
                                    -/
                                   /-
                                     C : Type u_1
                                     inst✝¹ : CategoryTheory.Category.{?u.163958, u_1} C
                                     inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                     S S₁✝ S₂✝ S₃ S₁ S₂ : CategoryTheory.ShortComplex C
                                     φ : Quiver.Hom S₁ S₂
                                     h₁ : S₁.RightHomologyData
                                     h₂ : S₂.RightHomologyData
                                     ψ : CategoryTheory.ShortComplex.RightHomologyMapData φ h₁ h₂
                                     ⊢ Eq (CategoryTheory.CategoryStruct.comp h₂.op.π ψ.φH.op).unop (CategoryTheory …
                                   -/
  commπ := Quiver.Hom.unop_inj (by simp)
                                   /-
                                     🎉 no goals
                                   -/


/-- A right homology map data for a morphism of short complexes in the opposite category
induces a left homology map data in the original category. -/
@[simps]
def RightHomologyMapData.unop {S₁ S₂ : ShortComplex Cᵒᵖ} {φ : S₁ ⟶ S₂}
    {h₁ : S₁.RightHomologyData} {h₂ : S₂.RightHomologyData}
    (ψ : RightHomologyMapData φ h₁ h₂) : LeftHomologyMapData (unopMap φ) h₂.unop h₁.unop where
  φK := ψ.φQ.unop
  φH := ψ.φH.unop
                                 /-
                                   C : Type u_1
                                   inst✝¹ : CategoryTheory.Category.{?u.166715, u_1} C
                                   inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                   S S₁✝ S₂✝ S₃ : CategoryTheory.ShortComplex C
                                   S₁ S₂ : CategoryTheory.ShortComplex (Opposite C)
                                   φ : Quiver.Hom S₁ S₂
                                   h₁ : S₁.RightHomologyData
                                   h₂ : S₂.RightHomologyData
                                   ψ : CategoryTheory.ShortComplex.RightHomologyMapData φ h₁ h₂
                                   ⊢ Eq (CategoryTheory.CategoryStruct.comp ψ.φQ.unop h₁.unop.i).op (CategoryTheo …
                                 -/
  commi := Quiver.Hom.op_inj (by simp)
                                 /-
                                   🎉 no goals
                                 -/
                                  /-
                                    C : Type u_1
                                    inst✝¹ : CategoryTheory.Category.{?u.166715, u_1} C
                                    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                    S S₁✝ S₂✝ S₃ : CategoryTheory.ShortComplex C
                                    S₁ S₂ : CategoryTheory.ShortComplex (Opposite C)
                                    φ : Quiver.Hom S₁ S₂
                                    h₁ : S₁.RightHomologyData
                                    h₂ : S₂.RightHomologyData
                                    ψ : CategoryTheory.ShortComplex.RightHomologyMapData φ h₁ h₂
                                    ⊢ Eq (CategoryTheory.CategoryStruct.comp h₂.unop.f' ψ.φQ.unop).op (CategoryThe …
                                  -/
  commf' := Quiver.Hom.op_inj (by simp)
                                  /-
                                    🎉 no goals
                                  -/
                                 /-
                                   C : Type u_1
                                   inst✝¹ : CategoryTheory.Category.{?u.166715, u_1} C
                                   inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                   S S₁✝ S₂✝ S₃ : CategoryTheory.ShortComplex C
                                   S₁ S₂ : CategoryTheory.ShortComplex (Opposite C)
                                   φ : Quiver.Hom S₁ S₂
                                   h₁ : S₁.RightHomologyData
                                   h₂ : S₂.RightHomologyData
                                   ψ : CategoryTheory.ShortComplex.RightHomologyMapData φ h₁ h₂
                                   ⊢ Eq (CategoryTheory.CategoryStruct.comp h₂.unop.π ψ.φH.unop).op (CategoryTheo …
                                 -/
  commπ := Quiver.Hom.op_inj (by simp)
                                 /-
                                   🎉 no goals
                                 -/


/-- The right homology in the opposite category of the opposite of a short complex identifies
to the left homology of this short complex. -/
noncomputable def rightHomologyOpIso [S.HasLeftHomology] :
    S.op.rightHomology ≅ Opposite.op S.leftHomology :=
  S.leftHomologyData.op.rightHomologyIso


/-- The left homology in the opposite category of the opposite of a short complex identifies
to the right homology of this short complex. -/
noncomputable def leftHomologyOpIso [S.HasRightHomology] :
    S.op.leftHomology ≅ Opposite.op S.rightHomology :=
  S.rightHomologyData.op.leftHomologyIso


/-- The opcycles in the opposite category of the opposite of a short complex identifies
to the cycles of this short complex. -/
noncomputable def opcyclesOpIso [S.HasLeftHomology] :
    S.op.opcycles ≅ Opposite.op S.cycles :=
  S.leftHomologyData.op.opcyclesIso


/-- The cycles in the opposite category of the opposite of a short complex identifies
to the opcycles of this short complex. -/
noncomputable def cyclesOpIso [S.HasRightHomology] :
    S.op.cycles ≅ Opposite.op S.opcycles :=
  S.rightHomologyData.op.cyclesIso


@[reassoc (attr := simp)]
lemma opcyclesOpIso_hom_toCycles_op [S.HasLeftHomology] :
    S.opcyclesOpIso.hom ≫ S.toCycles.op = S.op.fromOpcycles := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    inst✝ : S.HasLeftHomology
    ⊢ Eq (CategoryTheory.CategoryStruct.comp S.opcyclesOpIso.hom S.toCycles.op) S. …
  -/
  dsimp [opcyclesOpIso, toCycles]
  rw [← cancel_epi S.op.pOpcycles, p_fromOpcycles,
    RightHomologyData.pOpcycles_comp_opcyclesIso_hom_assoc,
    LeftHomologyData.op_p, ← op_comp, LeftHomologyData.f'_i, op_g]


@[reassoc (attr := simp)]
lemma fromOpcycles_op_cyclesOpIso_inv [S.HasRightHomology]:
    S.fromOpcycles.op ≫ S.cyclesOpIso.inv = S.op.toCycles := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    inst✝ : S.HasRightHomology
    ⊢ Eq (CategoryTheory.CategoryStruct.comp S.fromOpcycles.op S.cyclesOpIso.inv)  …
  -/
  dsimp [cyclesOpIso, fromOpcycles]
  rw [← cancel_mono S.op.iCycles, assoc, toCycles_i,
    LeftHomologyData.cyclesIso_inv_comp_iCycles, RightHomologyData.op_i,
    ← op_comp, RightHomologyData.p_g', op_f]


@[reassoc (attr := simp)]
lemma op_pOpcycles_opcyclesOpIso_hom [S.HasLeftHomology] :
    S.op.pOpcycles ≫ S.opcyclesOpIso.hom = S.iCycles.op := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    inst✝ : S.HasLeftHomology
    ⊢ Eq (CategoryTheory.CategoryStruct.comp S.op.pOpcycles S.opcyclesOpIso.hom) S …
  -/
  dsimp [opcyclesOpIso]
  rw [← S.leftHomologyData.op.p_comp_opcyclesIso_inv, assoc,
    Iso.inv_hom_id, comp_id]
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    inst✝ : S.HasLeftHomology
    ⊢ Eq S.leftHomologyData.op.p S.iCycles.op
  -/
  rfl
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma cyclesOpIso_inv_op_iCycles [S.HasRightHomology] :
    S.cyclesOpIso.inv ≫ S.op.iCycles = S.pOpcycles.op := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    inst✝ : S.HasRightHomology
    ⊢ Eq (CategoryTheory.CategoryStruct.comp S.cyclesOpIso.inv S.op.iCycles) S.pOp …
  -/
  dsimp [cyclesOpIso]
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    inst✝ : S.HasRightHomology
    ⊢ Eq (CategoryTheory.CategoryStruct.comp S.rightHomologyData.op.cyclesIso.inv  …
  -/
  rw [← S.rightHomologyData.op.cyclesIso_hom_comp_i, Iso.inv_hom_id_assoc]
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    inst✝ : S.HasRightHomology
    ⊢ Eq S.rightHomologyData.op.i S.pOpcycles.op
  -/
  rfl
  /-
    🎉 no goals
  -/


@[reassoc]
lemma opcyclesOpIso_hom_naturality (φ : S₁ ⟶ S₂)
    [S₁.HasLeftHomology] [S₂.HasLeftHomology] :
    opcyclesMap (opMap φ) ≫ (S₁.opcyclesOpIso).hom =
      S₂.opcyclesOpIso.hom ≫ (cyclesMap φ).op := by
  rw [← cancel_epi S₂.op.pOpcycles, p_opcyclesMap_assoc, opMap_τ₂,
    op_pOpcycles_opcyclesOpIso_hom, op_pOpcycles_opcyclesOpIso_hom_assoc, ← op_comp,
    ← op_comp, cyclesMap_i]


@[reassoc]
lemma opcyclesOpIso_inv_naturality (φ : S₁ ⟶ S₂)
    [S₁.HasLeftHomology] [S₂.HasLeftHomology] :
    (cyclesMap φ).op ≫ (S₁.opcyclesOpIso).inv =
      S₂.opcyclesOpIso.inv ≫ opcyclesMap (opMap φ) := by
  rw [← cancel_epi (S₂.opcyclesOpIso.hom), Iso.hom_inv_id_assoc,
    ← opcyclesOpIso_hom_naturality_assoc, Iso.hom_inv_id, comp_id]


@[reassoc]
lemma cyclesOpIso_inv_naturality (φ : S₁ ⟶ S₂)
    [S₁.HasRightHomology] [S₂.HasRightHomology] :
    (opcyclesMap φ).op ≫ (S₁.cyclesOpIso).inv =
      S₂.cyclesOpIso.inv ≫ cyclesMap (opMap φ) := by
  rw [← cancel_mono S₁.op.iCycles, assoc, assoc, cyclesOpIso_inv_op_iCycles, cyclesMap_i,
    cyclesOpIso_inv_op_iCycles_assoc, ← op_comp, p_opcyclesMap, op_comp, opMap_τ₂]


@[reassoc]
lemma cyclesOpIso_hom_naturality (φ : S₁ ⟶ S₂)
    [S₁.HasRightHomology] [S₂.HasRightHomology] :
    cyclesMap (opMap φ) ≫ (S₁.cyclesOpIso).hom =
      S₂.cyclesOpIso.hom ≫ (opcyclesMap φ).op := by
  rw [← cancel_mono (S₁.cyclesOpIso).inv, assoc, assoc, Iso.hom_inv_id, comp_id,
    cyclesOpIso_inv_naturality, Iso.hom_inv_id_assoc]


@[simp]
lemma leftHomologyMap'_op
    (φ : S₁ ⟶ S₂) (h₁ : S₁.LeftHomologyData) (h₂ : S₂.LeftHomologyData) :
    (leftHomologyMap' φ h₁ h₂).op = rightHomologyMap' (opMap φ) h₂.op h₁.op := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    h₁ : S₁.LeftHomologyData
    h₂ : S₂.LeftHomologyData
    ⊢ Eq (CategoryTheory.ShortComplex.leftHomologyMap' φ h₁ h₂).op (CategoryTheory …
  -/
  let γ : LeftHomologyMapData φ h₁ h₂ := leftHomologyMapData φ h₁ h₂
  simp only [γ.leftHomologyMap'_eq, γ.op.rightHomologyMap'_eq,
    LeftHomologyMapData.op_φH]


lemma leftHomologyMap_op (φ : S₁ ⟶ S₂) [S₁.HasLeftHomology] [S₂.HasLeftHomology] :
    (leftHomologyMap φ).op = S₂.rightHomologyOpIso.inv ≫ rightHomologyMap (opMap φ) ≫
      S₁.rightHomologyOpIso.hom := by
  dsimp [rightHomologyOpIso, RightHomologyData.rightHomologyIso, rightHomologyMap,
    leftHomologyMap]
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    inst✝¹ : S₁.HasLeftHomology
    inst✝ : S₂.HasLeftHomology
    ⊢ Eq (CategoryTheory.ShortComplex.leftHomologyMap' φ S₁.leftHomologyData S₂.le …
  -/
  simp only [← rightHomologyMap'_comp, comp_id, id_comp, leftHomologyMap'_op]
  /-
    🎉 no goals
  -/


@[simp]
lemma rightHomologyMap'_op
    (φ : S₁ ⟶ S₂) (h₁ : S₁.RightHomologyData) (h₂ : S₂.RightHomologyData) :
    (rightHomologyMap' φ h₁ h₂).op = leftHomologyMap' (opMap φ) h₂.op h₁.op := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    h₁ : S₁.RightHomologyData
    h₂ : S₂.RightHomologyData
    ⊢ Eq (CategoryTheory.ShortComplex.rightHomologyMap' φ h₁ h₂).op (CategoryTheor …
  -/
  let γ : RightHomologyMapData φ h₁ h₂ := rightHomologyMapData φ h₁ h₂
  simp only [γ.rightHomologyMap'_eq, γ.op.leftHomologyMap'_eq,
    RightHomologyMapData.op_φH]


lemma rightHomologyMap_op (φ : S₁ ⟶ S₂) [S₁.HasRightHomology] [S₂.HasRightHomology] :
    (rightHomologyMap φ).op = S₂.leftHomologyOpIso.inv ≫ leftHomologyMap (opMap φ) ≫
      S₁.leftHomologyOpIso.hom := by
  dsimp [leftHomologyOpIso, LeftHomologyData.leftHomologyIso, leftHomologyMap,
    rightHomologyMap]
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    inst✝¹ : S₁.HasRightHomology
    inst✝ : S₂.HasRightHomology
    ⊢ Eq (CategoryTheory.ShortComplex.rightHomologyMap' φ S₁.rightHomologyData S₂. …
  -/
  simp only [← leftHomologyMap'_comp, comp_id, id_comp, rightHomologyMap'_op]
  /-
    🎉 no goals
  -/


/-- If `φ : S₁ ⟶ S₂` is a morphism of short complexes such that `φ.τ₁` is epi, `φ.τ₂` is an iso
and `φ.τ₃` is mono, then a right homology data for `S₁` induces a right homology data for `S₂` with
the same `Q` and `H` fields. This is obtained by dualising `LeftHomologyData.ofEpiOfIsIsoOfMono'`.
The inverse construction is `ofEpiOfIsIsoOfMono'`. -/
noncomputable def ofEpiOfIsIsoOfMono : RightHomologyData S₂ := by
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{?u.203610, u_1} C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    S S₁ S₂ S₃ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    h : S₁.RightHomologyData
    inst✝² : CategoryTheory.Epi φ.τ₁
    inst✝¹ : CategoryTheory.IsIso φ.τ₂
    inst✝ : CategoryTheory.Mono φ.τ₃
    ⊢ S₂.RightHomologyData
  -/
  haveI : Epi (opMap φ).τ₁ := by dsimp; infer_instance
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{?u.203610, u_1} C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    S S₁ S₂ S₃ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    h : S₁.RightHomologyData
    inst✝² : CategoryTheory.Epi φ.τ₁
    inst✝¹ : CategoryTheory.IsIso φ.τ₂
    inst✝ : CategoryTheory.Mono φ.τ₃
    this : CategoryTheory.Epi (CategoryTheory.ShortComplex.opMap φ).τ₁
    ⊢ S₂.RightHomologyData
  -/
  haveI : IsIso (opMap φ).τ₂ := by dsimp; infer_instance
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{?u.203610, u_1} C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    S S₁ S₂ S₃ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    h : S₁.RightHomologyData
    inst✝² : CategoryTheory.Epi φ.τ₁
    inst✝¹ : CategoryTheory.IsIso φ.τ₂
    inst✝ : CategoryTheory.Mono φ.τ₃
    this✝ : CategoryTheory.Epi (CategoryTheory.ShortComplex.opMap φ).τ₁
    this : CategoryTheory.IsIso (CategoryTheory.ShortComplex.opMap φ).τ₂
    ⊢ S₂.RightHomologyData
  -/
  haveI : Mono (opMap φ).τ₃ := by dsimp; infer_instance
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{?u.203610, u_1} C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    S S₁ S₂ S₃ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    h : S₁.RightHomologyData
    inst✝² : CategoryTheory.Epi φ.τ₁
    inst✝¹ : CategoryTheory.IsIso φ.τ₂
    inst✝ : CategoryTheory.Mono φ.τ₃
    this✝¹ : CategoryTheory.Epi (CategoryTheory.ShortComplex.opMap φ).τ₁
    this✝ : CategoryTheory.IsIso (CategoryTheory.ShortComplex.opMap φ).τ₂
    this : CategoryTheory.Mono (CategoryTheory.ShortComplex.opMap φ).τ₃
    ⊢ S₂.RightHomologyData
  -/
  exact (LeftHomologyData.ofEpiOfIsIsoOfMono' (opMap φ) h.op).unop
  /-
    🎉 no goals
  -/


@[simp] lemma ofEpiOfIsIsoOfMono_Q : (ofEpiOfIsIsoOfMono φ h).Q = h.Q := rfl


@[simp] lemma ofEpiOfIsIsoOfMono_H : (ofEpiOfIsIsoOfMono φ h).H = h.H := rfl


@[simp] lemma ofEpiOfIsIsoOfMono_p : (ofEpiOfIsIsoOfMono φ h).p = inv φ.τ₂ ≫ h.p := by
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    h : S₁.RightHomologyData
    inst✝² : CategoryTheory.Epi φ.τ₁
    inst✝¹ : CategoryTheory.IsIso φ.τ₂
    inst✝ : CategoryTheory.Mono φ.τ₃
    ⊢ Eq (CategoryTheory.ShortComplex.RightHomologyData.ofEpiOfIsIsoOfMono φ h).p  …
  -/
  simp [ofEpiOfIsIsoOfMono, opMap]
  /-
    🎉 no goals
  -/


@[simp] lemma ofEpiOfIsIsoOfMono_ι : (ofEpiOfIsIsoOfMono φ h).ι = h.ι := rfl


@[simp] lemma ofEpiOfIsIsoOfMono_g' : (ofEpiOfIsIsoOfMono φ h).g' = h.g' ≫ φ.τ₃ := by
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    h : S₁.RightHomologyData
    inst✝² : CategoryTheory.Epi φ.τ₁
    inst✝¹ : CategoryTheory.IsIso φ.τ₂
    inst✝ : CategoryTheory.Mono φ.τ₃
    ⊢ Eq (CategoryTheory.ShortComplex.RightHomologyData.ofEpiOfIsIsoOfMono φ h).g' …
  -/
  simp [ofEpiOfIsIsoOfMono, opMap]
  /-
    🎉 no goals
  -/


/-- If `φ : S₁ ⟶ S₂` is a morphism of short complexes such that `φ.τ₁` is epi, `φ.τ₂` is an iso
and `φ.τ₃` is mono, then a right homology data for `S₂` induces a right homology data for `S₁` with
the same `Q` and `H` fields. This is obtained by dualising `LeftHomologyData.ofEpiOfIsIsoOfMono`.
The inverse construction is `ofEpiOfIsIsoOfMono`. -/
noncomputable def ofEpiOfIsIsoOfMono' : RightHomologyData S₁ := by
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{?u.214098, u_1} C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    S S₁ S₂ S₃ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    h : S₂.RightHomologyData
    inst✝² : CategoryTheory.Epi φ.τ₁
    inst✝¹ : CategoryTheory.IsIso φ.τ₂
    inst✝ : CategoryTheory.Mono φ.τ₃
    ⊢ S₁.RightHomologyData
  -/
  haveI : Epi (opMap φ).τ₁ := by dsimp; infer_instance
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{?u.214098, u_1} C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    S S₁ S₂ S₃ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    h : S₂.RightHomologyData
    inst✝² : CategoryTheory.Epi φ.τ₁
    inst✝¹ : CategoryTheory.IsIso φ.τ₂
    inst✝ : CategoryTheory.Mono φ.τ₃
    this : CategoryTheory.Epi (CategoryTheory.ShortComplex.opMap φ).τ₁
    ⊢ S₁.RightHomologyData
  -/
  haveI : IsIso (opMap φ).τ₂ := by dsimp; infer_instance
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{?u.214098, u_1} C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    S S₁ S₂ S₃ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    h : S₂.RightHomologyData
    inst✝² : CategoryTheory.Epi φ.τ₁
    inst✝¹ : CategoryTheory.IsIso φ.τ₂
    inst✝ : CategoryTheory.Mono φ.τ₃
    this✝ : CategoryTheory.Epi (CategoryTheory.ShortComplex.opMap φ).τ₁
    this : CategoryTheory.IsIso (CategoryTheory.ShortComplex.opMap φ).τ₂
    ⊢ S₁.RightHomologyData
  -/
  haveI : Mono (opMap φ).τ₃ := by dsimp; infer_instance
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{?u.214098, u_1} C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    S S₁ S₂ S₃ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    h : S₂.RightHomologyData
    inst✝² : CategoryTheory.Epi φ.τ₁
    inst✝¹ : CategoryTheory.IsIso φ.τ₂
    inst✝ : CategoryTheory.Mono φ.τ₃
    this✝¹ : CategoryTheory.Epi (CategoryTheory.ShortComplex.opMap φ).τ₁
    this✝ : CategoryTheory.IsIso (CategoryTheory.ShortComplex.opMap φ).τ₂
    this : CategoryTheory.Mono (CategoryTheory.ShortComplex.opMap φ).τ₃
    ⊢ S₁.RightHomologyData
  -/
  exact (LeftHomologyData.ofEpiOfIsIsoOfMono (opMap φ) h.op).unop
  /-
    🎉 no goals
  -/


@[simp] lemma ofEpiOfIsIsoOfMono'_Q : (ofEpiOfIsIsoOfMono' φ h).Q = h.Q := rfl


@[simp] lemma ofEpiOfIsIsoOfMono'_H : (ofEpiOfIsIsoOfMono' φ h).H = h.H := rfl


@[simp] lemma ofEpiOfIsIsoOfMono'_p : (ofEpiOfIsIsoOfMono' φ h).p = φ.τ₂ ≫ h.p := by
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    h : S₂.RightHomologyData
    inst✝² : CategoryTheory.Epi φ.τ₁
    inst✝¹ : CategoryTheory.IsIso φ.τ₂
    inst✝ : CategoryTheory.Mono φ.τ₃
    ⊢ Eq (CategoryTheory.ShortComplex.RightHomologyData.ofEpiOfIsIsoOfMono' φ h).p …
  -/
  simp [ofEpiOfIsIsoOfMono', opMap]
  /-
    🎉 no goals
  -/


@[simp] lemma ofEpiOfIsIsoOfMono'_ι : (ofEpiOfIsIsoOfMono' φ h).ι = h.ι := rfl


@[simp] lemma ofEpiOfIsIsoOfMono'_g'_τ₃ : (ofEpiOfIsIsoOfMono' φ h).g' ≫ φ.τ₃ = h.g' := by
  rw [← cancel_epi (ofEpiOfIsIsoOfMono' φ h).p, p_g'_assoc, ofEpiOfIsIsoOfMono'_p,
    assoc, p_g', φ.comm₂₃]


/-- If `e : S₁ ≅ S₂` is an isomorphism of short complexes and `h₁ : RightomologyData S₁`,
this is the right homology data for `S₂` deduced from the isomorphism. -/
noncomputable def ofIso (e : S₁ ≅ S₂) (h₁ : RightHomologyData S₁) : RightHomologyData S₂ :=
  h₁.ofEpiOfIsIsoOfMono e.hom


lemma hasRightHomology_of_epi_of_isIso_of_mono (φ : S₁ ⟶ S₂) [HasRightHomology S₁]
    [Epi φ.τ₁] [IsIso φ.τ₂] [Mono φ.τ₃] : HasRightHomology S₂ :=
  HasRightHomology.mk' (RightHomologyData.ofEpiOfIsIsoOfMono φ S₁.rightHomologyData)


lemma hasRightHomology_of_epi_of_isIso_of_mono' (φ : S₁ ⟶ S₂) [HasRightHomology S₂]
    [Epi φ.τ₁] [IsIso φ.τ₂] [Mono φ.τ₃] : HasRightHomology S₁ :=
HasRightHomology.mk' (RightHomologyData.ofEpiOfIsIsoOfMono' φ S₂.rightHomologyData)


lemma hasRightHomology_of_iso {S₁ S₂ : ShortComplex C}
    (e : S₁ ≅ S₂) [HasRightHomology S₁] : HasRightHomology S₂ :=
  hasRightHomology_of_epi_of_isIso_of_mono e.hom


/-- This right homology map data expresses compatibilities of the right homology data
constructed by `RightHomologyData.ofEpiOfIsIsoOfMono` -/
@[simps]
def ofEpiOfIsIsoOfMono (φ : S₁ ⟶ S₂) (h : RightHomologyData S₁)
    [Epi φ.τ₁] [IsIso φ.τ₂] [Mono φ.τ₃] :
    RightHomologyMapData φ h (RightHomologyData.ofEpiOfIsIsoOfMono φ h) where
  φQ := 𝟙 _
  φH := 𝟙 _


/-- This right homology map data expresses compatibilities of the right homology data
constructed by `RightHomologyData.ofEpiOfIsIsoOfMono'` -/
@[simps]
noncomputable def ofEpiOfIsIsoOfMono' (φ : S₁ ⟶ S₂) (h : RightHomologyData S₂)
    [Epi φ.τ₁] [IsIso φ.τ₂] [Mono φ.τ₃] :
    RightHomologyMapData φ (RightHomologyData.ofEpiOfIsIsoOfMono' φ h) h where
  φQ := 𝟙 _
  φH := 𝟙 _


instance (φ : S₁ ⟶ S₂) (h₁ : S₁.RightHomologyData) (h₂ : S₂.RightHomologyData)
    [Epi φ.τ₁] [IsIso φ.τ₂] [Mono φ.τ₃] :
    IsIso (rightHomologyMap' φ h₁ h₂) := by
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    S S₁ S₂ S₃ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    h₁ : S₁.RightHomologyData
    h₂ : S₂.RightHomologyData
    inst✝² : CategoryTheory.Epi φ.τ₁
    inst✝¹ : CategoryTheory.IsIso φ.τ₂
    inst✝ : CategoryTheory.Mono φ.τ₃
    ⊢ CategoryTheory.IsIso (CategoryTheory.ShortComplex.rightHomologyMap' φ h₁ h₂)
  -/
  let h₂' := RightHomologyData.ofEpiOfIsIsoOfMono φ h₁
  haveI : IsIso (rightHomologyMap' φ h₁ h₂') := by
    rw [(RightHomologyMapData.ofEpiOfIsIsoOfMono φ h₁).rightHomologyMap'_eq]
    dsimp
    infer_instance
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    S S₁ S₂ S₃ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    h₁ : S₁.RightHomologyData
    h₂ : S₂.RightHomologyData
    inst✝² : CategoryTheory.Epi φ.τ₁
    inst✝¹ : CategoryTheory.IsIso φ.τ₂
    inst✝ : CategoryTheory.Mono φ.τ₃
    h₂' : S₂.RightHomologyData := CategoryTheory.ShortComplex.RightHomologyData.of …
    this : CategoryTheory.IsIso (CategoryTheory.ShortComplex.rightHomologyMap' φ h …
    ⊢ CategoryTheory.IsIso (CategoryTheory.ShortComplex.rightHomologyMap' φ h₁ h₂)
  -/
  have eq := rightHomologyMap'_comp φ (𝟙 S₂) h₁ h₂' h₂
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    S S₁ S₂ S₃ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    h₁ : S₁.RightHomologyData
    h₂ : S₂.RightHomologyData
    inst✝² : CategoryTheory.Epi φ.τ₁
    inst✝¹ : CategoryTheory.IsIso φ.τ₂
    inst✝ : CategoryTheory.Mono φ.τ₃
    h₂' : S₂.RightHomologyData := CategoryTheory.ShortComplex.RightHomologyData.of …
    this : CategoryTheory.IsIso (CategoryTheory.ShortComplex.rightHomologyMap' φ h …
    eq : Eq (CategoryTheory.ShortComplex.rightHomologyMap' (CategoryTheory.Categor …
    ⊢ CategoryTheory.IsIso (CategoryTheory.ShortComplex.rightHomologyMap' φ h₁ h₂)
  -/
  rw [comp_id] at eq
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    S S₁ S₂ S₃ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    h₁ : S₁.RightHomologyData
    h₂ : S₂.RightHomologyData
    inst✝² : CategoryTheory.Epi φ.τ₁
    inst✝¹ : CategoryTheory.IsIso φ.τ₂
    inst✝ : CategoryTheory.Mono φ.τ₃
    h₂' : S₂.RightHomologyData := CategoryTheory.ShortComplex.RightHomologyData.of …
    this : CategoryTheory.IsIso (CategoryTheory.ShortComplex.rightHomologyMap' φ h …
    eq : Eq (CategoryTheory.ShortComplex.rightHomologyMap' φ h₁ h₂) (CategoryTheor …
    ⊢ CategoryTheory.IsIso (CategoryTheory.ShortComplex.rightHomologyMap' φ h₁ h₂)
  -/
  rw [eq]
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    S S₁ S₂ S₃ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    h₁ : S₁.RightHomologyData
    h₂ : S₂.RightHomologyData
    inst✝² : CategoryTheory.Epi φ.τ₁
    inst✝¹ : CategoryTheory.IsIso φ.τ₂
    inst✝ : CategoryTheory.Mono φ.τ₃
    h₂' : S₂.RightHomologyData := CategoryTheory.ShortComplex.RightHomologyData.of …
    this : CategoryTheory.IsIso (CategoryTheory.ShortComplex.rightHomologyMap' φ h …
    eq : Eq (CategoryTheory.ShortComplex.rightHomologyMap' φ h₁ h₂) (CategoryTheor …
    ⊢ CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp (CategoryTheory.Sho …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- If a morphism of short complexes `φ : S₁ ⟶ S₂` is such that `φ.τ₁` is epi, `φ.τ₂` is an iso,
and `φ.τ₃` is mono, then the induced morphism on right homology is an isomorphism. -/
instance (φ : S₁ ⟶ S₂) [S₁.HasRightHomology] [S₂.HasRightHomology]
    [Epi φ.τ₁] [IsIso φ.τ₂] [Mono φ.τ₃] :
    IsIso (rightHomologyMap φ) := by
  /-
    C : Type u_1
    inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C
    S S₁ S₂ S₃ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    inst✝⁴ : S₁.HasRightHomology
    inst✝³ : S₂.HasRightHomology
    inst✝² : CategoryTheory.Epi φ.τ₁
    inst✝¹ : CategoryTheory.IsIso φ.τ₂
    inst✝ : CategoryTheory.Mono φ.τ₃
    ⊢ CategoryTheory.IsIso (CategoryTheory.ShortComplex.rightHomologyMap φ)
  -/
  dsimp only [rightHomologyMap]
  /-
    C : Type u_1
    inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C
    S S₁ S₂ S₃ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    inst✝⁴ : S₁.HasRightHomology
    inst✝³ : S₂.HasRightHomology
    inst✝² : CategoryTheory.Epi φ.τ₁
    inst✝¹ : CategoryTheory.IsIso φ.τ₂
    inst✝ : CategoryTheory.Mono φ.τ₃
    ⊢ CategoryTheory.IsIso (CategoryTheory.ShortComplex.rightHomologyMap' φ S₁.rig …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- The opposite of the right homology functor is the left homology functor. -/
@[simps!]
noncomputable def rightHomologyFunctorOpNatIso :
    (rightHomologyFunctor C).op ≅ opFunctor C ⋙ leftHomologyFunctor Cᵒᵖ :=
  NatIso.ofComponents (fun S => (leftHomologyOpIso S.unop).symm)
        /-
          C : Type u_1
          inst✝⁵ : CategoryTheory.Category.{?u.237508, u_1} C
          inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
          S S₁ S₂ S₃ : CategoryTheory.ShortComplex C
          inst✝³ : CategoryTheory.Limits.HasKernels C
          inst✝² : CategoryTheory.Limits.HasCokernels C
          inst✝¹ : CategoryTheory.Limits.HasKernels (Opposite C)
          inst✝ : CategoryTheory.Limits.HasCokernels (Opposite C)
          ⊢ ∀ {X Y : Opposite (CategoryTheory.ShortComplex C)} (f : Quiver.Hom X Y), Eq  …
        -/
    (by simp [rightHomologyMap_op])
        /-
          🎉 no goals
        -/


/-- The opposite of the left homology functor is the right homology functor. -/
@[simps!]
noncomputable def leftHomologyFunctorOpNatIso :
    (leftHomologyFunctor C).op ≅ opFunctor C ⋙ rightHomologyFunctor Cᵒᵖ :=
  NatIso.ofComponents (fun S => (rightHomologyOpIso S.unop).symm)
        /-
          C : Type u_1
          inst✝⁵ : CategoryTheory.Category.{?u.242453, u_1} C
          inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
          S S₁ S₂ S₃ : CategoryTheory.ShortComplex C
          inst✝³ : CategoryTheory.Limits.HasKernels C
          inst✝² : CategoryTheory.Limits.HasCokernels C
          inst✝¹ : CategoryTheory.Limits.HasKernels (Opposite C)
          inst✝ : CategoryTheory.Limits.HasCokernels (Opposite C)
          ⊢ ∀ {X Y : Opposite (CategoryTheory.ShortComplex C)} (f : Quiver.Hom X Y), Eq  …
        -/
    (by simp [leftHomologyMap_op])
        /-
          🎉 no goals
        -/


/-- A morphism `k : S.X₂ ⟶ A` such that `S.f ≫ k = 0` descends to a morphism `S.opcycles ⟶ A`. -/
noncomputable def descOpcycles : S.opcycles ⟶ A :=
  S.rightHomologyData.descQ k hk


@[reassoc (attr := simp)]
lemma p_descOpcycles : S.pOpcycles ≫ S.descOpcycles k hk = k :=
  RightHomologyData.p_descQ _ k hk


@[reassoc]
lemma descOpcycles_comp {A' : C} (α : A ⟶ A') :
                                                         /-
                                                           C : Type u_1
                                                           inst✝² : CategoryTheory.Category.{?u.249907, u_1} C
                                                           inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
                                                           S S₁ S₂ S₃ : CategoryTheory.ShortComplex C
                                                           h : S.RightHomologyData
                                                           A : C
                                                           k : Quiver.Hom S.X₂ A
                                                           hk : Eq (CategoryTheory.CategoryStruct.comp S.f k) 0
                                                           inst✝ : S.HasRightHomology
                                                           A' : C
                                                           α : Quiver.Hom A A'
                                                           ⊢ Eq (CategoryTheory.CategoryStruct.comp S.f (CategoryTheory.CategoryStruct.co …
                                                         -/
    S.descOpcycles k hk ≫ α = S.descOpcycles (k ≫ α) (by rw [reassoc_of% hk, zero_comp]) := by
                                                         /-
                                                           🎉 no goals
                                                         -/
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    A : C
    k : Quiver.Hom S.X₂ A
    hk : Eq (CategoryTheory.CategoryStruct.comp S.f k) 0
    inst✝ : S.HasRightHomology
    A' : C
    α : Quiver.Hom A A'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (S.descOpcycles k hk) α) (S.descOpcyc …
  -/
  simp only [← cancel_epi S.pOpcycles, p_descOpcycles_assoc, p_descOpcycles]
  /-
    🎉 no goals
  -/


/-- Via `S.pOpcycles : S.X₂ ⟶ S.opcycles`, the object `S.opcycles` identifies to the
cokernel of `S.f : S.X₁ ⟶ S.X₂`. -/
noncomputable def opcyclesIsCokernel :
    IsColimit (CokernelCofork.ofπ S.pOpcycles S.f_pOpcycles) :=
  S.rightHomologyData.hp


/-- The canonical isomorphism `S.opcycles ≅ cokernel S.f`. -/
@[simps]
noncomputable def opcyclesIsoCokernel [HasCokernel S.f] : S.opcycles ≅ cokernel S.f where
                                             /-
                                               C : Type u_1
                                               inst✝³ : CategoryTheory.Category.{?u.253032, u_1} C
                                               inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
                                               S S₁ S₂ S₃ : CategoryTheory.ShortComplex C
                                               h : S.RightHomologyData
                                               A : C
                                               k : Quiver.Hom S.X₂ A
                                               hk : Eq (CategoryTheory.CategoryStruct.comp S.f k) 0
                                               inst✝¹ : S.HasRightHomology
                                               inst✝ : CategoryTheory.Limits.HasCokernel S.f
                                               ⊢ Eq (CategoryTheory.CategoryStruct.comp S.f (CategoryTheory.Limits.cokernel.π …
                                             -/
  hom := S.descOpcycles (cokernel.π S.f) (by simp)
                                             /-
                                               🎉 no goals
                                             -/
                                           /-
                                             C : Type u_1
                                             inst✝³ : CategoryTheory.Category.{?u.253032, u_1} C
                                             inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
                                             S S₁ S₂ S₃ : CategoryTheory.ShortComplex C
                                             h : S.RightHomologyData
                                             A : C
                                             k : Quiver.Hom S.X₂ A
                                             hk : Eq (CategoryTheory.CategoryStruct.comp S.f k) 0
                                             inst✝¹ : S.HasRightHomology
                                             inst✝ : CategoryTheory.Limits.HasCokernel S.f
                                             ⊢ Eq (CategoryTheory.CategoryStruct.comp S.f S.pOpcycles) 0
                                           -/
  inv := cokernel.desc S.f S.pOpcycles (by simp)
                                           /-
                                             🎉 no goals
                                           -/


/-- The morphism `S.rightHomology ⟶ A` obtained from a morphism `k : S.X₂ ⟶ A`
such that `S.f ≫ k = 0.` -/
@[simp]
noncomputable def descRightHomology : S.rightHomology ⟶ A :=
  S.rightHomologyι ≫ S.descOpcycles k hk


@[reassoc]
lemma rightHomologyι_descOpcycles_π_eq_zero_of_boundary (x : S.X₃ ⟶ A) (hx : k = S.g ≫ x) :
                                            /-
                                              C : Type u_1
                                              inst✝² : CategoryTheory.Category.{?u.257211, u_1} C
                                              inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
                                              S S₁ S₂ S₃ : CategoryTheory.ShortComplex C
                                              h : S.RightHomologyData
                                              A : C
                                              k : Quiver.Hom S.X₂ A
                                              hk : Eq (CategoryTheory.CategoryStruct.comp S.f k) 0
                                              inst✝ : S.HasRightHomology
                                              x : Quiver.Hom S.X₃ A
                                              hx : Eq k (CategoryTheory.CategoryStruct.comp S.g x)
                                              ⊢ Eq (CategoryTheory.CategoryStruct.comp S.f k) 0
                                            -/
    S.rightHomologyι ≫ S.descOpcycles k (by rw [hx, S.zero_assoc, zero_comp]) = 0 :=
                                            /-
                                              🎉 no goals
                                            -/
  RightHomologyData.ι_descQ_eq_zero_of_boundary _ k x hx


@[reassoc (attr := simp)]
lemma rightHomologyι_comp_fromOpcycles :
    S.rightHomologyι ≫ S.fromOpcycles = 0 :=
                                                                    /-
                                                                      C : Type u_1
                                                                      inst✝² : CategoryTheory.Category.{u_2, u_1} C
                                                                      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
                                                                      S : CategoryTheory.ShortComplex C
                                                                      inst✝ : S.HasRightHomology
                                                                      ⊢ Eq S.g (CategoryTheory.CategoryStruct.comp S.g (CategoryTheory.CategoryStruc …
                                                                    -/
  S.rightHomologyι_descOpcycles_π_eq_zero_of_boundary S.g (𝟙 _) (by rw [comp_id])
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


/-- Via `S.rightHomologyι : S.rightHomology ⟶ S.opcycles`, the object `S.rightHomology` identifies
to the kernel of `S.fromOpcycles : S.opcycles ⟶ S.X₃`. -/
noncomputable def rightHomologyIsKernel :
    IsLimit (KernelFork.ofι S.rightHomologyι S.rightHomologyι_comp_fromOpcycles) :=
  S.rightHomologyData.hι


@[reassoc (attr := simp)]
lemma opcyclesMap_comp_descOpcycles (φ : S₁ ⟶ S) [S₁.HasRightHomology] :
    opcyclesMap φ ≫ S.descOpcycles k hk =
                                     /-
                                       C : Type u_1
                                       inst✝³ : CategoryTheory.Category.{?u.261871, u_1} C
                                       inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
                                       S S₁ S₂ S₃ : CategoryTheory.ShortComplex C
                                       h : S.RightHomologyData
                                       A : C
                                       k : Quiver.Hom S.X₂ A
                                       hk : Eq (CategoryTheory.CategoryStruct.comp S.f k) 0
                                       inst✝¹ : S.HasRightHomology
                                       φ : Quiver.Hom S₁ S
                                       inst✝ : S₁.HasRightHomology
                                       ⊢ Eq (CategoryTheory.CategoryStruct.comp S₁.f (CategoryTheory.CategoryStruct.c …
                                     -/
      S₁.descOpcycles (φ.τ₂ ≫ k) (by rw [← φ.comm₁₂_assoc, hk, comp_zero]) := by
                                     /-
                                       🎉 no goals
                                     -/
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    S S₁ : CategoryTheory.ShortComplex C
    A : C
    k : Quiver.Hom S.X₂ A
    hk : Eq (CategoryTheory.CategoryStruct.comp S.f k) 0
    inst✝¹ : S.HasRightHomology
    φ : Quiver.Hom S₁ S
    inst✝ : S₁.HasRightHomology
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.ShortComplex.opcycles …
  -/
  simp only [← cancel_epi (S₁.pOpcycles), p_opcyclesMap_assoc, p_descOpcycles]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma RightHomologyData.opcyclesIso_inv_comp_descOpcycles :
    h.opcyclesIso.inv ≫ S.descOpcycles k hk = h.descQ k hk := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    h : S.RightHomologyData
    A : C
    k : Quiver.Hom S.X₂ A
    hk : Eq (CategoryTheory.CategoryStruct.comp S.f k) 0
    inst✝ : S.HasRightHomology
    ⊢ Eq (CategoryTheory.CategoryStruct.comp h.opcyclesIso.inv (S.descOpcycles k h …
  -/
  simp only [← cancel_epi h.p, p_comp_opcyclesIso_inv_assoc, p_descOpcycles, p_descQ]
  /-
    🎉 no goals
  -/


@[simp]
lemma RightHomologyData.opcyclesIso_hom_comp_descQ :
    h.opcyclesIso.hom ≫ h.descQ k hk = S.descOpcycles k hk := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    h : S.RightHomologyData
    A : C
    k : Quiver.Hom S.X₂ A
    hk : Eq (CategoryTheory.CategoryStruct.comp S.f k) 0
    inst✝ : S.HasRightHomology
    ⊢ Eq (CategoryTheory.CategoryStruct.comp h.opcyclesIso.hom (h.descQ k hk)) (S. …
  -/
  rw [← h.opcyclesIso_inv_comp_descOpcycles, Iso.hom_inv_id_assoc]
  /-
    🎉 no goals
  -/


lemma hasCokernel [S.HasRightHomology] : HasCokernel S.f :=
  ⟨⟨⟨_, S.rightHomologyData.hp⟩⟩⟩


lemma hasKernel [S.HasRightHomology] [HasCokernel S.f] :
    HasKernel (cokernel.desc S.f S.g S.zero) := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    inst✝¹ : S.HasRightHomology
    inst✝ : CategoryTheory.Limits.HasCokernel S.f
    ⊢ CategoryTheory.Limits.HasKernel (CategoryTheory.Limits.cokernel.desc S.f S.g …
  -/
  let h := S.rightHomologyData
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    inst✝¹ : S.HasRightHomology
    inst✝ : CategoryTheory.Limits.HasCokernel S.f
    h : S.RightHomologyData := S.rightHomologyData
    ⊢ CategoryTheory.Limits.HasKernel (CategoryTheory.Limits.cokernel.desc S.f S.g …
  -/
  haveI : HasLimit (parallelPair h.g' 0) := ⟨⟨⟨_, h.hι'⟩⟩⟩
  let e : parallelPair (cokernel.desc S.f S.g S.zero) 0 ≅ parallelPair h.g' 0 :=
    parallelPair.ext (IsColimit.coconePointUniqueUpToIso (colimit.isColimit _) h.hp)
      (Iso.refl _) (coequalizer.hom_ext (by simp)) (by aesop_cat)
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ShortComplex C
    inst✝¹ : S.HasRightHomology
    inst✝ : CategoryTheory.Limits.HasCokernel S.f
    h : S.RightHomologyData := S.rightHomologyData
    this : CategoryTheory.Limits.HasLimit (CategoryTheory.Limits.parallelPair h.g' …
    e : CategoryTheory.Iso (CategoryTheory.Limits.parallelPair (CategoryTheory.Lim …
    ⊢ CategoryTheory.Limits.HasKernel (CategoryTheory.Limits.cokernel.desc S.f S.g …
  -/
  exact hasLimitOfIso e.symm
  /-
    🎉 no goals
  -/


/-- The right homology of a short complex `S` identifies to the kernel of the canonical
morphism `cokernel S.f ⟶ S.X₃`. -/
noncomputable def rightHomologyIsoKernelDesc [S.HasRightHomology] [HasCokernel S.f]
    [HasKernel (cokernel.desc S.f S.g S.zero)] :
    S.rightHomology ≅ kernel (cokernel.desc S.f S.g S.zero) :=
  (RightHomologyData.ofHasCokernelOfHasKernel S).rightHomologyIso


lemma isIso_opcyclesMap'_of_isIso_of_epi (φ : S₁ ⟶ S₂) (h₂ : IsIso φ.τ₂) (h₁ : Epi φ.τ₁)
    (h₁ : S₁.RightHomologyData) (h₂ : S₂.RightHomologyData) :
    IsIso (opcyclesMap' φ h₁ h₂) := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    h₂✝ : CategoryTheory.IsIso φ.τ₂
    h₁✝ : CategoryTheory.Epi φ.τ₁
    h₁ : S₁.RightHomologyData
    h₂ : S₂.RightHomologyData
    ⊢ CategoryTheory.IsIso (CategoryTheory.ShortComplex.opcyclesMap' φ h₁ h₂)
  -/
  refine ⟨h₂.descQ (inv φ.τ₂ ≫ h₁.p) ?_, ?_, ?_⟩
    /-
      case refine_1
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      S₁ S₂ : CategoryTheory.ShortComplex C
      φ : Quiver.Hom S₁ S₂
      h₂✝ : CategoryTheory.IsIso φ.τ₂
      h₁✝ : CategoryTheory.Epi φ.τ₁
      h₁ : S₁.RightHomologyData
      h₂ : S₂.RightHomologyData
      ⊢ Eq (CategoryTheory.CategoryStruct.comp S₂.f (CategoryTheory.CategoryStruct.c …
    -/
  · simp only [← cancel_epi φ.τ₁, comp_zero, φ.comm₁₂_assoc, IsIso.hom_inv_id_assoc, h₁.wp]
    /-
      🎉 no goals
    -/
  · simp only [← cancel_epi h₁.p, p_opcyclesMap'_assoc, h₂.p_descQ,
      IsIso.hom_inv_id_assoc, comp_id]
  · simp only [← cancel_epi h₂.p, h₂.p_descQ_assoc, assoc, p_opcyclesMap',
      IsIso.inv_hom_id_assoc, comp_id]


lemma isIso_opcyclesMap_of_isIso_of_epi' (φ : S₁ ⟶ S₂) (h₂ : IsIso φ.τ₂) (h₁ : Epi φ.τ₁)
    [S₁.HasRightHomology] [S₂.HasRightHomology] :
    IsIso (opcyclesMap φ) :=
  isIso_opcyclesMap'_of_isIso_of_epi φ h₂ h₁ _ _


instance isIso_opcyclesMap_of_isIso_of_epi (φ : S₁ ⟶ S₂) [IsIso φ.τ₂] [Epi φ.τ₁]
    [S₁.HasRightHomology] [S₂.HasRightHomology] :
    IsIso (opcyclesMap φ) :=
  isIso_opcyclesMap_of_isIso_of_epi' φ inferInstance inferInstance


