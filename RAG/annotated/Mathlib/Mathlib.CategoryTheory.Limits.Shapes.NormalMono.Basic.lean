/-- A normal monomorphism is a morphism which is the kernel of some morphism. -/
class NormalMono (f : X ⟶ Y) where
  Z : C -- Porting note: violates naming convention but can't think of a better one
  g : Y ⟶ Z
  w : f ≫ g = 0
  isLimit : IsLimit (KernelFork.ofι f w)


/-- If `F` is an equivalence and `F.map f` is a normal mono, then `f` is a normal mono. -/
def equivalenceReflectsNormalMono {D : Type u₂} [Category.{v₁} D] [HasZeroMorphisms D] (F : C ⥤ D)
    [F.IsEquivalence] {X Y : C} {f : X ⟶ Y} (hf : NormalMono (F.map f)) : NormalMono f where
  Z := F.objPreimage hf.Z
  g := F.preimage (hf.g ≫ (F.objObjPreimageIso hf.Z).inv)
  w := F.map_injective <| by
    have reassoc' {W : D} (h : hf.Z ⟶ W) : F.map f ≫ hf.g ≫ h = 0 ≫ h := by
      rw [← Category.assoc, eq_whisker hf.w]
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      X✝ Y✝ : C
      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₁, u₂} D
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms D
      F : CategoryTheory.Functor C D
      inst✝ : F.IsEquivalence
      X Y : C
      f : Quiver.Hom X Y
      hf : CategoryTheory.NormalMono (F.map f)
      reassoc' : ∀ {W : D} (h : Quiver.Hom (CategoryTheory.NormalMono.Z (F.map f)) W …
      ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp f (F.preimage (CategoryTheory. …
    -/
    simp [reassoc']
    /-
      🎉 no goals
    -/
  isLimit := isLimitOfReflects F <|
    IsLimit.ofConeEquiv (Cones.postcomposeEquivalence (compNatIso F)) <|
      (IsLimit.ofIsoLimit (IsKernel.ofCompIso _ _ (F.objObjPreimageIso hf.Z) (by
        /-
          C : Type u₁
          inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
          X✝ Y✝ : C
          inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
          D : Type u₂
          inst✝² : CategoryTheory.Category.{v₁, u₂} D
          inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms D
          F : CategoryTheory.Functor C D
          inst✝ : F.IsEquivalence
          X Y : C
          f : Quiver.Hom X Y
          hf : CategoryTheory.NormalMono (F.map f)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (F.preimage (CategoryTheory.Ca …
        -/
        simp only [Functor.map_preimage, Category.assoc, Iso.inv_hom_id, Category.comp_id])
        /-
          🎉 no goals
        -/
                                                /-
                                                  C : Type u₁
                                                  inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
                                                  X✝ Y✝ : C
                                                  inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
                                                  D : Type u₂
                                                  inst✝² : CategoryTheory.Category.{v₁, u₂} D
                                                  inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms D
                                                  F : CategoryTheory.Functor C D
                                                  inst✝ : F.IsEquivalence
                                                  X Y : C
                                                  f : Quiver.Hom X Y
                                                  hf : CategoryTheory.NormalMono (F.map f)
                                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Iso.refl (CategoryThe …
                                                -/
        hf.isLimit)) (Fork.ext (Iso.refl _) (by simp [compNatIso, Fork.ι]))
                                                /-
                                                  🎉 no goals
                                                -/


/-- Every normal monomorphism is a regular monomorphism. -/
instance (priority := 100) NormalMono.regularMono (f : X ⟶ Y) [I : NormalMono f] : RegularMono f :=
  { I with
    left := I.g
    right := 0
            /-
              C : Type u₁
              inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
              X Y : C
              inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
              f : Quiver.Hom X Y
              I : CategoryTheory.NormalMono f
              ⊢ Eq (CategoryTheory.CategoryStruct.comp f CategoryTheory.NormalMono.g) (Categ …
            -/
    w := by simpa using I.w }
            /-
              🎉 no goals
            -/


/-- If `f` is a normal mono, then any map `k : W ⟶ Y` such that `k ≫ normal_mono.g = 0` induces
    a morphism `l : W ⟶ X` such that `l ≫ f = k`. -/
def NormalMono.lift' {W : C} (f : X ⟶ Y) [hf : NormalMono f] (k : W ⟶ Y) (h : k ≫ hf.g = 0) :
    { l : W ⟶ X // l ≫ f = k } :=
  KernelFork.IsLimit.lift' NormalMono.isLimit _ h


/-- The second leg of a pullback cone is a normal monomorphism if the right component is too.

See also `pullback.sndOfMono` for the basic monomorphism version, and
`normalOfIsPullbackFstOfNormal` for the flipped version.
-/
def normalOfIsPullbackSndOfNormal {P Q R S : C} {f : P ⟶ Q} {g : P ⟶ R} {h : Q ⟶ S} {k : R ⟶ S}
    [hn : NormalMono h] (comm : f ≫ h = g ≫ k) (t : IsLimit (PullbackCone.mk _ _ comm)) :
    NormalMono g where
  Z := hn.Z
  g := k ≫ hn.g
  w := by
    have reassoc' {W : C} (h' : S ⟶ W) : f ≫ h ≫ h' = g ≫ k ≫ h' := by
      simp only [← Category.assoc, eq_whisker comm]
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      X Y : C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      P Q R S : C
      f : Quiver.Hom P Q
      g : Quiver.Hom P R
      h : Quiver.Hom Q S
      k : Quiver.Hom R S
      hn : CategoryTheory.NormalMono h
      comm : Eq (CategoryTheory.CategoryStruct.comp f h) (CategoryTheory.CategoryStr …
      t : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk f g c …
      reassoc' : ∀ {W : C} (h' : Quiver.Hom S W), Eq (CategoryTheory.CategoryStruct. …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp g (CategoryTheory.CategoryStruct.comp …
    -/
    rw [← reassoc', hn.w, HasZeroMorphisms.comp_zero]
    /-
      🎉 no goals
    -/
  isLimit := by
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      X Y : C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      P Q R S : C
      f : Quiver.Hom P Q
      g : Quiver.Hom P R
      h : Quiver.Hom Q S
      k : Quiver.Hom R S
      hn : CategoryTheory.NormalMono h
      comm : Eq (CategoryTheory.CategoryStruct.comp f h) (CategoryTheory.CategoryStr …
      t : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk f g c …
      ⊢ CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι g ⋯)
    -/
    letI gr := regularOfIsPullbackSndOfRegular comm t
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      X Y : C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      P Q R S : C
      f : Quiver.Hom P Q
      g : Quiver.Hom P R
      h : Quiver.Hom Q S
      k : Quiver.Hom R S
      hn : CategoryTheory.NormalMono h
      comm : Eq (CategoryTheory.CategoryStruct.comp f h) (CategoryTheory.CategoryStr …
      t : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk f g c …
      gr : CategoryTheory.RegularMono g := CategoryTheory.regularOfIsPullbackSndOfRe …
      ⊢ CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι g ⋯)
    -/
    have q := (HasZeroMorphisms.comp_zero k hn.Z).symm
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      X Y : C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      P Q R S : C
      f : Quiver.Hom P Q
      g : Quiver.Hom P R
      h : Quiver.Hom Q S
      k : Quiver.Hom R S
      hn : CategoryTheory.NormalMono h
      comm : Eq (CategoryTheory.CategoryStruct.comp f h) (CategoryTheory.CategoryStr …
      t : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk f g c …
      gr : CategoryTheory.RegularMono g := CategoryTheory.regularOfIsPullbackSndOfRe …
      q : Eq 0 (CategoryTheory.CategoryStruct.comp k 0)
      ⊢ CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι g ⋯)
    -/
    convert gr.isLimit
    /-
      🎉 no goals
    -/


/-- The first leg of a pullback cone is a normal monomorphism if the left component is too.

See also `pullback.fstOfMono` for the basic monomorphism version, and
`normalOfIsPullbackSndOfNormal` for the flipped version.
-/
def normalOfIsPullbackFstOfNormal {P Q R S : C} {f : P ⟶ Q} {g : P ⟶ R} {h : Q ⟶ S} {k : R ⟶ S}
    [NormalMono k] (comm : f ≫ h = g ≫ k) (t : IsLimit (PullbackCone.mk _ _ comm)) :
    NormalMono f :=
  normalOfIsPullbackSndOfNormal comm.symm (PullbackCone.flipIsLimit t)


/-- A normal mono category is a category in which every monomorphism is normal. -/
class NormalMonoCategory where
  normalMonoOfMono : ∀ {X Y : C} (f : X ⟶ Y) [Mono f], NormalMono f


/-- In a category in which every monomorphism is normal, we can express every monomorphism as
    a kernel. This is not an instance because it would create an instance loop. -/
def normalMonoOfMono [NormalMonoCategory C] (f : X ⟶ Y) [Mono f] : NormalMono f :=
  NormalMonoCategory.normalMonoOfMono _


instance (priority := 100) regularMonoCategoryOfNormalMonoCategory [NormalMonoCategory C] :
    RegularMonoCategory C where
  regularMonoOfMono f _ := by
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      X Y : C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝ : CategoryTheory.NormalMonoCategory C
      X✝ Y✝ : C
      f : Quiver.Hom X✝ Y✝
      x✝ : CategoryTheory.Mono f
      ⊢ CategoryTheory.RegularMono f
    -/
    haveI := normalMonoOfMono f
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      X Y : C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝ : CategoryTheory.NormalMonoCategory C
      X✝ Y✝ : C
      f : Quiver.Hom X✝ Y✝
      x✝ : CategoryTheory.Mono f
      this : CategoryTheory.NormalMono f
      ⊢ CategoryTheory.RegularMono f
    -/
    infer_instance
    /-
      🎉 no goals
    -/


/-- A normal epimorphism is a morphism which is the cokernel of some morphism. -/
class NormalEpi (f : X ⟶ Y) where
  W : C
  g : W ⟶ X
  w : g ≫ f = 0
  isColimit : IsColimit (CokernelCofork.ofπ f w)


/-- If `F` is an equivalence and `F.map f` is a normal epi, then `f` is a normal epi. -/
def equivalenceReflectsNormalEpi {D : Type u₂} [Category.{v₁} D] [HasZeroMorphisms D] (F : C ⥤ D)
    [F.IsEquivalence] {X Y : C} {f : X ⟶ Y} (hf : NormalEpi (F.map f)) : NormalEpi f where
  W := F.objPreimage hf.W
  g := F.preimage ((F.objObjPreimageIso hf.W).hom ≫ hf.g)
                             /-
                               C : Type u₁
                               inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
                               X✝ Y✝ : C
                               inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
                               D : Type u₂
                               inst✝² : CategoryTheory.Category.{v₁, u₂} D
                               inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms D
                               F : CategoryTheory.Functor C D
                               inst✝ : F.IsEquivalence
                               X Y : C
                               f : Quiver.Hom X Y
                               hf : CategoryTheory.NormalEpi (F.map f)
                               ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp (F.preimage (CategoryTheory.Ca …
                             -/
  w := F.map_injective <| by simp [hf.w]
                             /-
                               🎉 no goals
                             -/
  isColimit := isColimitOfReflects F <|
    IsColimit.ofCoconeEquiv (Cocones.precomposeEquivalence (compNatIso F).symm) <|
      (IsColimit.ofIsoColimit
                                                                      /-
                                                                        C : Type u₁
                                                                        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
                                                                        X✝ Y✝ : C
                                                                        inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
                                                                        D : Type u₂
                                                                        inst✝² : CategoryTheory.Category.{v₁, u₂} D
                                                                        inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms D
                                                                        F : CategoryTheory.Functor C D
                                                                        inst✝ : F.IsEquivalence
                                                                        X Y : C
                                                                        f : Quiver.Hom X Y
                                                                        hf : CategoryTheory.NormalEpi (F.map f)
                                                                        ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.objObjPreimageIso (CategoryTheory. …
                                                                      -/
        (IsCokernel.ofIsoComp _ _ (F.objObjPreimageIso hf.W).symm (by simp) hf.isColimit)
                                                                      /-
                                                                        🎉 no goals
                                                                      -/
                                       /-
                                         C : Type u₁
                                         inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
                                         X✝ Y✝ : C
                                         inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
                                         D : Type u₂
                                         inst✝² : CategoryTheory.Category.{v₁, u₂} D
                                         inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms D
                                         F : CategoryTheory.Functor C D
                                         inst✝ : F.IsEquivalence
                                         X Y : C
                                         f : Quiver.Hom X Y
                                         hf : CategoryTheory.NormalEpi (F.map f)
                                         ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π (Cate …
                                       -/
          (Cofork.ext (Iso.refl _) (by simp [compNatIso, Cofork.π])))
                                       /-
                                         🎉 no goals
                                       -/


/-- Every normal epimorphism is a regular epimorphism. -/
instance (priority := 100) NormalEpi.regularEpi (f : X ⟶ Y) [I : NormalEpi f] : RegularEpi f :=
  { I with
    left := I.g
    right := 0
            /-
              C : Type u₁
              inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
              X Y : C
              inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
              f : Quiver.Hom X Y
              I : CategoryTheory.NormalEpi f
              ⊢ Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.NormalEpi.g f) (Catego …
            -/
    w := by simpa using I.w }
            /-
              🎉 no goals
            -/


/-- If `f` is a normal epi, then every morphism `k : X ⟶ W` satisfying `NormalEpi.g ≫ k = 0`
    induces `l : Y ⟶ W` such that `f ≫ l = k`. -/
def NormalEpi.desc' {W : C} (f : X ⟶ Y) [nef : NormalEpi f] (k : X ⟶ W) (h : nef.g ≫ k = 0) :
    { l : Y ⟶ W // f ≫ l = k } :=
  CokernelCofork.IsColimit.desc' NormalEpi.isColimit _ h


/-- The second leg of a pushout cocone is a normal epimorphism if the right component is too.

See also `pushout.sndOfEpi` for the basic epimorphism version, and
`normalOfIsPushoutFstOfNormal` for the flipped version.
-/
def normalOfIsPushoutSndOfNormal {P Q R S : C} {f : P ⟶ Q} {g : P ⟶ R} {h : Q ⟶ S} {k : R ⟶ S}
    [gn : NormalEpi g] (comm : f ≫ h = g ≫ k) (t : IsColimit (PushoutCocone.mk _ _ comm)) :
    NormalEpi h where
  W := gn.W
  g := gn.g ≫ f
  w := by
    have reassoc' {W : C} (h' : R ⟶ W) :  gn.g ≫ g ≫ h' = 0 ≫ h' := by
      rw [← Category.assoc, eq_whisker gn.w]
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      X Y : C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      P Q R S : C
      f : Quiver.Hom P Q
      g : Quiver.Hom P R
      h : Quiver.Hom Q S
      k : Quiver.Hom R S
      gn : CategoryTheory.NormalEpi g
      comm : Eq (CategoryTheory.CategoryStruct.comp f h) (CategoryTheory.CategoryStr …
      t : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.PushoutCocone.mk h  …
      reassoc' : ∀ {W : C} (h' : Quiver.Hom R W), Eq (CategoryTheory.CategoryStruct. …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp C …
    -/
    rw [Category.assoc, comm, reassoc', zero_comp]
    /-
      🎉 no goals
    -/
  isColimit := by
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      X Y : C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      P Q R S : C
      f : Quiver.Hom P Q
      g : Quiver.Hom P R
      h : Quiver.Hom Q S
      k : Quiver.Hom R S
      gn : CategoryTheory.NormalEpi g
      comm : Eq (CategoryTheory.CategoryStruct.comp f h) (CategoryTheory.CategoryStr …
      t : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.PushoutCocone.mk h  …
      ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ h ⋯)
    -/
    letI hn := regularOfIsPushoutSndOfRegular comm t
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      X Y : C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      P Q R S : C
      f : Quiver.Hom P Q
      g : Quiver.Hom P R
      h : Quiver.Hom Q S
      k : Quiver.Hom R S
      gn : CategoryTheory.NormalEpi g
      comm : Eq (CategoryTheory.CategoryStruct.comp f h) (CategoryTheory.CategoryStr …
      t : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.PushoutCocone.mk h  …
      hn : CategoryTheory.RegularEpi h := CategoryTheory.regularOfIsPushoutSndOfRegu …
      ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ h ⋯)
    -/
    have q := (@zero_comp _ _ _ gn.W _ _ f).symm
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      X Y : C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      P Q R S : C
      f : Quiver.Hom P Q
      g : Quiver.Hom P R
      h : Quiver.Hom Q S
      k : Quiver.Hom R S
      gn : CategoryTheory.NormalEpi g
      comm : Eq (CategoryTheory.CategoryStruct.comp f h) (CategoryTheory.CategoryStr …
      t : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.PushoutCocone.mk h  …
      hn : CategoryTheory.RegularEpi h := CategoryTheory.regularOfIsPushoutSndOfRegu …
      q : Eq 0 (CategoryTheory.CategoryStruct.comp 0 f)
      ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ h ⋯)
    -/
    convert hn.isColimit
    /-
      🎉 no goals
    -/


/-- The first leg of a pushout cocone is a normal epimorphism if the left component is too.

See also `pushout.fstOfEpi` for the basic epimorphism version, and
`normalOfIsPushoutSndOfNormal` for the flipped version.
-/
def normalOfIsPushoutFstOfNormal {P Q R S : C} {f : P ⟶ Q} {g : P ⟶ R} {h : Q ⟶ S} {k : R ⟶ S}
    [NormalEpi f] (comm : f ≫ h = g ≫ k) (t : IsColimit (PushoutCocone.mk _ _ comm)) :
    NormalEpi k :=
  normalOfIsPushoutSndOfNormal comm.symm (PushoutCocone.flipIsColimit t)


/-- A normal mono becomes a normal epi in the opposite category. -/
def normalEpiOfNormalMonoUnop {X Y : Cᵒᵖ} (f : X ⟶ Y) (m : NormalMono f.unop) : NormalEpi f where
  W := op m.Z
  g := m.g.op
  w := congrArg Quiver.Hom.op m.w
  isColimit :=
    CokernelCofork.IsColimit.ofπ _ _
      (fun g' w' =>
        (KernelFork.IsLimit.lift' m.isLimit g'.unop (congrArg Quiver.Hom.unop w')).1.op)
      (fun g' w' =>
        congrArg Quiver.Hom.op
          (KernelFork.IsLimit.lift' m.isLimit g'.unop (congrArg Quiver.Hom.unop w')).2)
      (by
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          X✝ Y✝ : C
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          X Y : Opposite C
          f : Quiver.Hom X Y
          m : CategoryTheory.NormalMono f.unop
          ⊢ ∀ {Z' : Opposite C} (g' : Quiver.Hom X Z') (eq' : Eq (CategoryTheory.Categor …
        -/
        rintro Z' g' w' m' rfl
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          X✝ Y✝ : C
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          X Y : Opposite C
          f : Quiver.Hom X Y
          m : CategoryTheory.NormalMono f.unop
          Z' : Opposite C
          m' : Quiver.Hom Y Z'
          w' : Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.NormalMono.g.op (Ca …
          ⊢ Eq m' ((fun {Z'} g' w' => (↑(CategoryTheory.Limits.KernelFork.IsLimit.lift'  …
        -/
        apply Quiver.Hom.unop_inj
        /-
          case a
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          X✝ Y✝ : C
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          X Y : Opposite C
          f : Quiver.Hom X Y
          m : CategoryTheory.NormalMono f.unop
          Z' : Opposite C
          m' : Quiver.Hom Y Z'
          w' : Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.NormalMono.g.op (Ca …
          ⊢ Eq m'.unop ((fun {Z'} g' w' => (↑(CategoryTheory.Limits.KernelFork.IsLimit.l …
        -/
        apply m.isLimit.uniq (KernelFork.ofι (m'.unop ≫ f.unop) _) m'.unop
        /-
          case a
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          X✝ Y✝ : C
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          X Y : Opposite C
          f : Quiver.Hom X Y
          m : CategoryTheory.NormalMono f.unop
          Z' : Opposite C
          m' : Quiver.Hom Y Z'
          w' : Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.NormalMono.g.op (Ca …
          ⊢ ∀ (j : CategoryTheory.Limits.WalkingParallelPair), Eq (CategoryTheory.Catego …
        -/
                             /-
                               🎉 no goals
                             -/
        rintro (⟨⟩ | ⟨⟩) <;> simp)
                             /-
                               🎉 no goals
                             -/


/-- A normal epi becomes a normal mono in the opposite category. -/
def normalMonoOfNormalEpiUnop {X Y : Cᵒᵖ} (f : X ⟶ Y) (m : NormalEpi f.unop) : NormalMono f where
  Z := op m.W
  g := m.g.op
  w := congrArg Quiver.Hom.op m.w
  isLimit :=
    KernelFork.IsLimit.ofι _ _
      (fun g' w' =>
        (CokernelCofork.IsColimit.desc' m.isColimit g'.unop (congrArg Quiver.Hom.unop w')).1.op)
      (fun g' w' =>
        congrArg Quiver.Hom.op
          (CokernelCofork.IsColimit.desc' m.isColimit g'.unop (congrArg Quiver.Hom.unop w')).2)
      (by
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          X✝ Y✝ : C
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          X Y : Opposite C
          f : Quiver.Hom X Y
          m : CategoryTheory.NormalEpi f.unop
          ⊢ ∀ {W' : Opposite C} (g' : Quiver.Hom W' Y) (eq' : Eq (CategoryTheory.Categor …
        -/
        rintro Z' g' w' m' rfl
        /-
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          X✝ Y✝ : C
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          X Y : Opposite C
          f : Quiver.Hom X Y
          m : CategoryTheory.NormalEpi f.unop
          Z' : Opposite C
          m' : Quiver.Hom Z' X
          w' : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.com …
          ⊢ Eq m' ((fun {W'} g' w' => (↑(CategoryTheory.Limits.CokernelCofork.IsColimit. …
        -/
        apply Quiver.Hom.unop_inj
        /-
          case a
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          X✝ Y✝ : C
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          X Y : Opposite C
          f : Quiver.Hom X Y
          m : CategoryTheory.NormalEpi f.unop
          Z' : Opposite C
          m' : Quiver.Hom Z' X
          w' : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.com …
          ⊢ Eq m'.unop ((fun {W'} g' w' => (↑(CategoryTheory.Limits.CokernelCofork.IsCol …
        -/
        apply m.isColimit.uniq (CokernelCofork.ofπ (f.unop ≫ m'.unop) _) m'.unop
        /-
          case a
          C : Type u₁
          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
          X✝ Y✝ : C
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          X Y : Opposite C
          f : Quiver.Hom X Y
          m : CategoryTheory.NormalEpi f.unop
          Z' : Opposite C
          m' : Quiver.Hom Z' X
          w' : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.com …
          ⊢ ∀ (j : CategoryTheory.Limits.WalkingParallelPair), Eq (CategoryTheory.Catego …
        -/
                             /-
                               🎉 no goals
                             -/
        rintro (⟨⟩ | ⟨⟩) <;> simp)
                             /-
                               🎉 no goals
                             -/


/-- A normal epi category is a category in which every epimorphism is normal. -/
class NormalEpiCategory where
  normalEpiOfEpi : ∀ {X Y : C} (f : X ⟶ Y) [Epi f], NormalEpi f


/-- In a category in which every epimorphism is normal, we can express every epimorphism as
    a kernel. This is not an instance because it would create an instance loop. -/
def normalEpiOfEpi [NormalEpiCategory C] (f : X ⟶ Y) [Epi f] : NormalEpi f :=
  NormalEpiCategory.normalEpiOfEpi _


instance (priority := 100) regularEpiCategoryOfNormalEpiCategory [NormalEpiCategory C] :
    RegularEpiCategory C where
  regularEpiOfEpi f _ := by
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      X Y : C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝ : CategoryTheory.NormalEpiCategory C
      X✝ Y✝ : C
      f : Quiver.Hom X✝ Y✝
      x✝ : CategoryTheory.Epi f
      ⊢ CategoryTheory.RegularEpi f
    -/
    haveI := normalEpiOfEpi f
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      X Y : C
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝ : CategoryTheory.NormalEpiCategory C
      X✝ Y✝ : C
      f : Quiver.Hom X✝ Y✝
      x✝ : CategoryTheory.Epi f
      this : CategoryTheory.NormalEpi f
      ⊢ CategoryTheory.RegularEpi f
    -/
    infer_instance
    /-
      🎉 no goals
    -/


