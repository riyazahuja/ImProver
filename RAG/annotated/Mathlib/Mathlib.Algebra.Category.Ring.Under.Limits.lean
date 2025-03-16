/-- The canonical fan on `P : ι → Under R` given by `∀ i, P i`. -/
def piFan : Fan P :=
  Fan.mk (Under.mk <| ofHom <| Pi.ringHom (fun i ↦ (P i).hom.hom))
             /-
               R S : CommRingCat
               inst✝ : Algebra ↑R ↑S
               ι : Type u
               P : ι → CategoryTheory.Under R
               i : ι
               ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Under.mk (CommRingCat …
             -/
    (fun i ↦ Under.homMk (ofHom <| Pi.evalRingHom _ i))
             /-
               🎉 no goals
             -/


/-- The canonical fan is limiting. -/
def piFanIsLimit : IsLimit (piFan P) :=
  isLimitOfReflects (Under.forget R) <|
    (isLimitMapConeFanMkEquiv (Under.forget R) P _).symm <|
      CommRingCat.piFanIsLimit (fun i ↦ (P i).right)


variable (S) in
/-- The fan on `i ↦ S ⊗[R] P i` given by `S ⊗[R] ∀ i, P i` -/
def tensorProductFan : Fan (fun i ↦ mkUnder S (S ⊗[R] (P i).right)) :=
  Fan.mk (mkUnder S <| S ⊗[R] ∀ i, (P i).right)
    (fun i ↦ AlgHom.toUnder <|
      Algebra.TensorProduct.map (AlgHom.id S S) (Pi.evalAlgHom R (fun j ↦ (P j).right) i))


variable (S) in
/-- The fan on `i ↦ S ⊗[R] P i` given by `∀ i, S ⊗[R] P i` -/
def tensorProductFan' : Fan (fun i ↦ mkUnder S (S ⊗[R] (P i).right)) :=
  Fan.mk (mkUnder S <| ∀ i, S ⊗[R] (P i).right)
    (fun i ↦ AlgHom.toUnder <| Pi.evalAlgHom S _ i)


/-- The two fans on `i ↦ S ⊗[R] P i` agree if `ι` is finite. -/
def tensorProductFanIso [Fintype ι] [DecidableEq ι] :
    tensorProductFan S P ≅ tensorProductFan' S P :=
  Fan.ext (Algebra.TensorProduct.piRight R S _ _).toUnder <| fun i ↦ by
    /-
      R S : CommRingCat
      inst✝² : Algebra ↑R ↑S
      ι : Type u
      P : ι → CategoryTheory.Under R
      inst✝¹ : Fintype ι
      inst✝ : DecidableEq ι
      i : ι
      ⊢ Eq ((CommRingCat.Under.tensorProductFan S P).proj i) (CategoryTheory.Categor …
    -/
    dsimp only [tensorProductFan, Fan.mk_pt, fan_mk_proj, tensorProductFan']
    /-
      R S : CommRingCat
      inst✝² : Algebra ↑R ↑S
      ι : Type u
      P : ι → CategoryTheory.Under R
      inst✝¹ : Fintype ι
      inst✝ : DecidableEq ι
      i : ι
      ⊢ Eq (Algebra.TensorProduct.map (AlgHom.id ↑S ↑S) (Pi.evalAlgHom (↑R) (fun j = …
    -/
    apply CommRingCat.mkUnder_ext
    /-
      case h
      R S : CommRingCat
      inst✝² : Algebra ↑R ↑S
      ι : Type u
      P : ι → CategoryTheory.Under R
      inst✝¹ : Fintype ι
      inst✝ : DecidableEq ι
      i : ι
      ⊢ ∀ (a : TensorProduct (↑R) (↑S) ((i : ι) → ↑(P i).right)), Eq ((Algebra.Tenso …
    -/
    intro c
    /-
      case h
      R S : CommRingCat
      inst✝² : Algebra ↑R ↑S
      ι : Type u
      P : ι → CategoryTheory.Under R
      inst✝¹ : Fintype ι
      inst✝ : DecidableEq ι
      i : ι
      c : TensorProduct (↑R) (↑S) ((i : ι) → ↑(P i).right)
      ⊢ Eq ((Algebra.TensorProduct.map (AlgHom.id ↑S ↑S) (Pi.evalAlgHom (↑R) (fun j  …
    -/
    induction c
    · simp only [AlgHom.toUnder_right, map_zero, Under.comp_right, comp_apply,
        AlgEquiv.toUnder_hom_right_apply, Pi.evalAlgHom_apply, Pi.zero_apply]
    · simp only [AlgHom.toUnder_right, Algebra.TensorProduct.map_tmul, AlgHom.coe_id, id_eq,
        Pi.evalAlgHom_apply, Under.comp_right, comp_apply, AlgEquiv.toUnder_hom_right_apply,
        Algebra.TensorProduct.piRight_tmul]
      /-
        case h.add
        R S : CommRingCat
        inst✝² : Algebra ↑R ↑S
        ι : Type u
        P : ι → CategoryTheory.Under R
        inst✝¹ : Fintype ι
        inst✝ : DecidableEq ι
        i : ι
        x✝ y✝ : TensorProduct (↑R) (↑S) ((i : ι) → ↑(P i).right)
        a✝¹ : Eq ((Algebra.TensorProduct.map (AlgHom.id ↑S ↑S) (Pi.evalAlgHom (↑R) (fu …
        a✝ : Eq ((Algebra.TensorProduct.map (AlgHom.id ↑S ↑S) (Pi.evalAlgHom (↑R) (fun …
        ⊢ Eq ((Algebra.TensorProduct.map (AlgHom.id ↑S ↑S) (Pi.evalAlgHom (↑R) (fun j  …
      -/
    · simp_all
      /-
        🎉 no goals
      -/


open Classical in
/-- The fan on `i ↦ S ⊗[R] P i` given by `S ⊗[R] ∀ i, P i` is limiting if `ι` is finite. -/
def tensorProductFanIsLimit [Finite ι] : IsLimit (tensorProductFan S P) :=
  letI : Fintype ι := Fintype.ofFinite ι
  (IsLimit.equivIsoLimit (tensorProductFanIso P)).symm (Under.piFanIsLimit _)


/-- `tensorProd R S` preserves the limit of the canonical fan on `P`. -/
noncomputable -- marked noncomputable for performance (only)
def piFanTensorProductIsLimit [Finite ι] : IsLimit ((tensorProd R S).mapCone (Under.piFan P)) :=
  (isLimitMapConeFanMkEquiv (tensorProd R S) P _).symm <| tensorProductFanIsLimit P


instance (J : Type u) [Finite J] (f : J → Under R) :
    PreservesLimit (Discrete.functor f) (tensorProd R S) :=
  let c : Fan _ := Under.piFan f
  have hc : IsLimit c := Under.piFanIsLimit f
  preservesLimit_of_preserves_limit_cone hc (piFanTensorProductIsLimit f)


instance (J : Type) [Finite J] :
    PreservesLimitsOfShape (Discrete J) (tensorProd R S) :=
  let J' : Type u := ULift.{u} J
  have : PreservesLimitsOfShape (Discrete J') (tensorProd R S) :=
    preservesLimitsOfShape_of_discrete (tensorProd R S)
  let e : Discrete J' ≌ Discrete J := Discrete.equivalence Equiv.ulift
  preservesLimitsOfShape_of_equiv e (R.tensorProd S)


instance : PreservesFiniteProducts (tensorProd R S) where
  preserves J := { }


lemma equalizer_comp {A B : Under R} (f g : A ⟶ B) :
    (AlgHom.equalizer (toAlgHom f) (toAlgHom g)).val.toUnder ≫ f =
    (AlgHom.equalizer (toAlgHom f) (toAlgHom g)).val.toUnder ≫ g := by
  /-
    R : CommRingCat
    A B : CategoryTheory.Under R
    f g : Quiver.Hom A B
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgHom.equalizer (CommRingCat.toAlgH …
  -/
  ext (a : AlgHom.equalizer (toAlgHom f) (toAlgHom g))
  /-
    case h
    R : CommRingCat
    A B : CategoryTheory.Under R
    f g : Quiver.Hom A B
    a : Subtype fun x => Membership.mem (AlgHom.equalizer (CommRingCat.toAlgHom f) …
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp (AlgHom.equalizer (CommRingCat.toAlg …
  -/
  exact a.property
  /-
    🎉 no goals
  -/


/-- The canonical fork on `f g : A ⟶ B` given by the equalizer. -/
def equalizerFork {A B : Under R} (f g : A ⟶ B) :
    Fork f g :=
  Fork.ofι ((AlgHom.equalizer (toAlgHom f) (toAlgHom g)).val.toUnder)
        /-
          R S : CommRingCat
          inst✝ : Algebra ↑R ↑S
          A B : CategoryTheory.Under R
          f g : Quiver.Hom A B
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgHom.equalizer (CommRingCat.toAlgH …
        -/
    (by rw [equalizer_comp])
        /-
          🎉 no goals
        -/


@[simp]
lemma equalizerFork_ι {A B : Under R} (f g : A ⟶ B) :
    (Under.equalizerFork f g).ι = (AlgHom.equalizer (toAlgHom f) (toAlgHom g)).val.toUnder := rfl


/-- Variant of `Under.equalizerFork'` for algebra maps. This is definitionally equal to
`Under.equalizerFork` but this is costly in applications. -/
def equalizerFork' {A B : Type u} [CommRing A] [CommRing B] [Algebra R A] [Algebra R B]
    (f g : A →ₐ[R] B) :
    Fork f.toUnder g.toUnder :=
                                                      /-
                                                        R S : CommRingCat
                                                        inst✝⁴ : Algebra ↑R ↑S
                                                        A B : Type u
                                                        inst✝³ : CommRing A
                                                        inst✝² : CommRing B
                                                        inst✝¹ : Algebra (↑R) A
                                                        inst✝ : Algebra (↑R) B
                                                        f g : AlgHom (↑R) A B
                                                        ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgHom.equalizer f g).val.toUnder f. …
                                                      -/
  Fork.ofι ((AlgHom.equalizer f g).val.toUnder) <| by ext a; exact a.property
                                                             /-
                                                               🎉 no goals
                                                             -/


@[simp]
lemma equalizerFork'_ι {A B : Type u} [CommRing A] [CommRing B] [Algebra R A] [Algebra R B]
    (f g : A →ₐ[R] B) :
    (Under.equalizerFork' f g).ι = (AlgHom.equalizer f g).val.toUnder := rfl


/-- The canonical fork on `f g : A ⟶ B` is limiting. -/
-- marked noncomputable for performance (only)
noncomputable def equalizerForkIsLimit {A B : Under R} (f g : A ⟶ B) :
    IsLimit (Under.equalizerFork f g) :=
  isLimitOfReflects (Under.forget R) <|
    (isLimitMapConeForkEquiv (Under.forget R) (equalizer_comp f g)).invFun <|
      CommRingCat.equalizerForkIsLimit f.right g.right


/-- Variant of `Under.equalizerForkIsLimit` for algebra maps. -/
def equalizerFork'IsLimit {A B : Type u} [CommRing A] [CommRing B] [Algebra R A]
    [Algebra R B] (f g : A →ₐ[R] B) :
    IsLimit (Under.equalizerFork' f g) :=
  Under.equalizerForkIsLimit f.toUnder g.toUnder


/-- The fork on `𝟙 ⊗[R] f` and `𝟙 ⊗[R] g` given by `S ⊗[R] eq(f, g)`. -/
def tensorProdEqualizer {A B : Under R} (f g : A ⟶ B) :
    Fork ((tensorProd R S).map f) ((tensorProd R S).map g) :=
  Fork.ofι
    ((tensorProd R S).map ((AlgHom.equalizer (toAlgHom f) (toAlgHom g)).val.toUnder)) <| by
    /-
      R S : CommRingCat
      inst✝ : Algebra ↑R ↑S
      A B : CategoryTheory.Under R
      f g : Quiver.Hom A B
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((R.tensorProd S).map (AlgHom.equaliz …
    -/
    rw [← Functor.map_comp, equalizer_comp, Functor.map_comp]
    /-
      🎉 no goals
    -/


@[simp]
lemma tensorProdEqualizer_ι {A B : Under R} (f g : A ⟶ B) :
    (tensorProdEqualizer f g).ι = (tensorProd R S).map
      ((AlgHom.equalizer (toAlgHom f) (toAlgHom g)).val.toUnder) :=
  rfl


/-- If `S` is `R`-flat, `S ⊗[R] eq(f, g)` is isomorphic to `eq(𝟙 ⊗[R] f, 𝟙 ⊗[R] g)`. -/
-- marked noncomputable for performance (only)
noncomputable def equalizerForkTensorProdIso [Module.Flat R S] {A B : Under R} (f g : A ⟶ B) :
    tensorProdEqualizer f g ≅ Under.equalizerFork'
        (Algebra.TensorProduct.map (AlgHom.id S S) (toAlgHom f))
        (Algebra.TensorProduct.map (AlgHom.id S S) (toAlgHom g)) :=
  Fork.ext (AlgHom.tensorEqualizerEquiv S S (toAlgHom f) (toAlgHom g)).toUnder <| by
    /-
      R S : CommRingCat
      inst✝¹ : Algebra ↑R ↑S
      inst✝ : Module.Flat ↑R ↑S
      A B : CategoryTheory.Under R
      f g : Quiver.Hom A B
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgHom.tensorEqualizerEquiv (↑S) (↑S …
    -/
    ext
    /-
      case h.hf.a
      R S : CommRingCat
      inst✝¹ : Algebra ↑R ↑S
      inst✝ : Module.Flat ↑R ↑S
      A B : CategoryTheory.Under R
      f g : Quiver.Hom A B
      x✝ : ↑(CommRingCat.Under.tensorProdEqualizer f g).pt.right
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (AlgHom.tensorEqualizerEquiv (↑S) (↑ …
    -/
    apply AlgHom.coe_tensorEqualizer
    /-
      🎉 no goals
    -/


/-- If `S` is `R`-flat, `tensorProd R S` preserves the equalizer of `f` and `g`. -/
noncomputable -- marked noncomputable for performance (only)
def tensorProdMapEqualizerForkIsLimit [Module.Flat R S] {A B : Under R} (f g : A ⟶ B) :
    IsLimit ((tensorProd R S).mapCone <| Under.equalizerFork f g) :=
  (isLimitMapConeForkEquiv (tensorProd R S) _).symm <|
    (IsLimit.equivIsoLimit (equalizerForkTensorProdIso f g).symm) <|
    Under.equalizerFork'IsLimit _ _


instance [Module.Flat R S] {A B : Under R} (f g : A ⟶ B) :
    PreservesLimit (parallelPair f g) (tensorProd R S) :=
  let c : Fork f g := Under.equalizerFork f g
  let hc : IsLimit c := Under.equalizerForkIsLimit f g
  let hc' : IsLimit ((tensorProd R S).mapCone c) :=
    tensorProdMapEqualizerForkIsLimit f g
  preservesLimit_of_preserves_limit_cone hc hc'


instance [Module.Flat R S] : PreservesLimitsOfShape WalkingParallelPair (tensorProd R S) where
  preservesLimit {K} :=
    preservesLimit_of_iso_diagram _ (diagramIsoParallelPair K).symm


instance [Module.Flat R S] : PreservesFiniteLimits (tensorProd R S) :=
  preservesFiniteLimits_of_preservesEqualizers_and_finiteProducts (tensorProd R S)


/-- `Under.pushout f` preserves finite products. -/
instance : PreservesFiniteProducts (Under.pushout f) where
  preserves _ :=
    letI : Algebra R S := f.hom.toAlgebra
    preservesLimitsOfShape_of_natIso (tensorProdIsoPushout R S)


/-- `Under.pushout f` preserves finite limits if `f` is flat. -/
lemma preservesFiniteLimits_of_flat (hf : RingHom.Flat f.hom) :
    PreservesFiniteLimits (Under.pushout f) where
  preservesFiniteLimits _ :=
    letI : Algebra R S := f.hom.toAlgebra
    haveI : Module.Flat R S := hf
    preservesLimitsOfShape_of_natIso (tensorProdIsoPushout R S)


