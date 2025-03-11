/-- A functor between preadditive categories which preserves kernels preserves that an
arbitrary binary fan is a limit.
-/
def isLimitMapConeBinaryFanOfPreservesKernels {X Y Z : C} (π₁ : Z ⟶ X) (π₂ : Z ⟶ Y)
    [PreservesLimit (parallelPair π₂ 0) F] (i : IsLimit (BinaryFan.mk π₁ π₂)) :
    IsLimit (F.mapCone (BinaryFan.mk π₁ π₂)) := by
  /-
    C : Type u₁
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁴ : CategoryTheory.Preadditive C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    inst✝² : CategoryTheory.Preadditive D
    F : CategoryTheory.Functor C D
    inst✝¹ : F.PreservesZeroMorphisms
    X Y Z : C
    π₁ : Quiver.Hom Z X
    π₂ : Quiver.Hom Z Y
    inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.parallelPa …
    i : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.BinaryFan.mk π₁ π₂)
    ⊢ CategoryTheory.Limits.IsLimit (F.mapCone (CategoryTheory.Limits.BinaryFan.mk …
  -/
  let bc := BinaryBicone.ofLimitCone i
  /-
    C : Type u₁
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁴ : CategoryTheory.Preadditive C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    inst✝² : CategoryTheory.Preadditive D
    F : CategoryTheory.Functor C D
    inst✝¹ : F.PreservesZeroMorphisms
    X Y Z : C
    π₁ : Quiver.Hom Z X
    π₂ : Quiver.Hom Z Y
    inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.parallelPa …
    i : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.BinaryFan.mk π₁ π₂)
    bc : CategoryTheory.Limits.BinaryBicone X Y := CategoryTheory.Limits.BinaryBic …
    ⊢ CategoryTheory.Limits.IsLimit (F.mapCone (CategoryTheory.Limits.BinaryFan.mk …
  -/
  let presf : PreservesLimit (parallelPair bc.snd 0) F := by simpa
  /-
    C : Type u₁
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁴ : CategoryTheory.Preadditive C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    inst✝² : CategoryTheory.Preadditive D
    F : CategoryTheory.Functor C D
    inst✝¹ : F.PreservesZeroMorphisms
    X Y Z : C
    π₁ : Quiver.Hom Z X
    π₂ : Quiver.Hom Z Y
    inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.parallelPa …
    i : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.BinaryFan.mk π₁ π₂)
    bc : CategoryTheory.Limits.BinaryBicone X Y := CategoryTheory.Limits.BinaryBic …
    presf : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.parallelPa …
    ⊢ CategoryTheory.Limits.IsLimit (F.mapCone (CategoryTheory.Limits.BinaryFan.mk …
  -/
  let hf : IsLimit bc.sndKernelFork := BinaryBicone.isLimitSndKernelFork i
  exact (isLimitMapConeBinaryFanEquiv F π₁ π₂).invFun
    (BinaryBicone.isBilimitOfKernelInl (F.mapBinaryBicone bc)
    (isLimitMapConeForkEquiv' F bc.inl_snd (isLimitOfPreserves F hf))).isLimit


/-- A kernel preserving functor between preadditive categories preserves any pair being a limit. -/
lemma preservesBinaryProduct_of_preservesKernels
    [∀ {X Y} (f : X ⟶ Y), PreservesLimit (parallelPair f 0) F] {X Y : C} :
    PreservesLimit (pair X Y) F where
  preserves {c} hc :=
    ⟨IsLimit.ofIsoLimit
      (isLimitMapConeBinaryFanOfPreservesKernels F _ _ (IsLimit.ofIsoLimit hc (isoBinaryFanMk c)))
      ((Cones.functoriality _ F).mapIso (isoBinaryFanMk c).symm)⟩


/-- A kernel preserving functor between preadditive categories preserves binary products. -/
lemma preservesBinaryProducts_of_preservesKernels
    [∀ {X Y} (f : X ⟶ Y), PreservesLimit (parallelPair f 0) F] :
  PreservesLimitsOfShape (Discrete WalkingPair) F where
    preservesLimit := preservesLimit_of_iso_diagram F (diagramIsoPair _).symm


/-- A functor between preadditive categories preserves the equalizer of two
morphisms if it preserves all kernels. -/
lemma preservesEqualizer_of_preservesKernels
    [∀ {X Y} (f : X ⟶ Y), PreservesLimit (parallelPair f 0) F]
    {X Y : C} (f g : X ⟶ Y) : PreservesLimit (parallelPair f g) F := by
  /-
    C : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁵ : CategoryTheory.Preadditive C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    inst✝³ : CategoryTheory.Preadditive D
    F : CategoryTheory.Functor C D
    inst✝² : F.PreservesZeroMorphisms
    inst✝¹ : CategoryTheory.Limits.HasBinaryBiproducts C
    inst✝ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.Limits.PreservesLimit …
    X Y : C
    f g : Quiver.Hom X Y
    ⊢ CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.parallelPair f g …
  -/
  letI := preservesBinaryBiproducts_of_preservesBinaryProducts F
  /-
    C : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁵ : CategoryTheory.Preadditive C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    inst✝³ : CategoryTheory.Preadditive D
    F : CategoryTheory.Functor C D
    inst✝² : F.PreservesZeroMorphisms
    inst✝¹ : CategoryTheory.Limits.HasBinaryBiproducts C
    inst✝ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.Limits.PreservesLimit …
    X Y : C
    f g : Quiver.Hom X Y
    this : CategoryTheory.Limits.PreservesBinaryBiproducts F := CategoryTheory.Lim …
    ⊢ CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.parallelPair f g …
  -/
  haveI := additive_of_preservesBinaryBiproducts F
  /-
    C : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁵ : CategoryTheory.Preadditive C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    inst✝³ : CategoryTheory.Preadditive D
    F : CategoryTheory.Functor C D
    inst✝² : F.PreservesZeroMorphisms
    inst✝¹ : CategoryTheory.Limits.HasBinaryBiproducts C
    inst✝ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.Limits.PreservesLimit …
    X Y : C
    f g : Quiver.Hom X Y
    this✝ : CategoryTheory.Limits.PreservesBinaryBiproducts F := CategoryTheory.Li …
    this : F.Additive
    ⊢ CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.parallelPair f g …
  -/
  constructor; intro c i
  /-
    case preserves
    C : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁵ : CategoryTheory.Preadditive C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    inst✝³ : CategoryTheory.Preadditive D
    F : CategoryTheory.Functor C D
    inst✝² : F.PreservesZeroMorphisms
    inst✝¹ : CategoryTheory.Limits.HasBinaryBiproducts C
    inst✝ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.Limits.PreservesLimit …
    X Y : C
    f g : Quiver.Hom X Y
    this✝ : CategoryTheory.Limits.PreservesBinaryBiproducts F := CategoryTheory.Li …
    this : F.Additive
    c : CategoryTheory.Limits.Cone (CategoryTheory.Limits.parallelPair f g)
    i : CategoryTheory.Limits.IsLimit c
    ⊢ Nonempty (CategoryTheory.Limits.IsLimit (F.mapCone c))
  -/
  let c' := isLimitKernelForkOfFork (i.ofIsoLimit (Fork.isoForkOfι c))
  /-
    case preserves
    C : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁵ : CategoryTheory.Preadditive C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    inst✝³ : CategoryTheory.Preadditive D
    F : CategoryTheory.Functor C D
    inst✝² : F.PreservesZeroMorphisms
    inst✝¹ : CategoryTheory.Limits.HasBinaryBiproducts C
    inst✝ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.Limits.PreservesLimit …
    X Y : C
    f g : Quiver.Hom X Y
    this✝ : CategoryTheory.Limits.PreservesBinaryBiproducts F := CategoryTheory.Li …
    this : F.Additive
    c : CategoryTheory.Limits.Cone (CategoryTheory.Limits.parallelPair f g)
    i : CategoryTheory.Limits.IsLimit c
    c' : CategoryTheory.Limits.IsLimit (CategoryTheory.Preadditive.kernelForkOfFor …
    ⊢ Nonempty (CategoryTheory.Limits.IsLimit (F.mapCone c))
  -/
  dsimp only [kernelForkOfFork_ofι] at c'
  /-
    case preserves
    C : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁵ : CategoryTheory.Preadditive C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    inst✝³ : CategoryTheory.Preadditive D
    F : CategoryTheory.Functor C D
    inst✝² : F.PreservesZeroMorphisms
    inst✝¹ : CategoryTheory.Limits.HasBinaryBiproducts C
    inst✝ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.Limits.PreservesLimit …
    X Y : C
    f g : Quiver.Hom X Y
    this✝ : CategoryTheory.Limits.PreservesBinaryBiproducts F := CategoryTheory.Li …
    this : F.Additive
    c : CategoryTheory.Limits.Cone (CategoryTheory.Limits.parallelPair f g)
    i : CategoryTheory.Limits.IsLimit c
    c' : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι (Cate …
    ⊢ Nonempty (CategoryTheory.Limits.IsLimit (F.mapCone c))
  -/
  let iFc := isLimitForkMapOfIsLimit' F _ c'
  /-
    case preserves
    C : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁵ : CategoryTheory.Preadditive C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    inst✝³ : CategoryTheory.Preadditive D
    F : CategoryTheory.Functor C D
    inst✝² : F.PreservesZeroMorphisms
    inst✝¹ : CategoryTheory.Limits.HasBinaryBiproducts C
    inst✝ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.Limits.PreservesLimit …
    X Y : C
    f g : Quiver.Hom X Y
    this✝ : CategoryTheory.Limits.PreservesBinaryBiproducts F := CategoryTheory.Li …
    this : F.Additive
    c : CategoryTheory.Limits.Cone (CategoryTheory.Limits.parallelPair f g)
    i : CategoryTheory.Limits.IsLimit c
    c' : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι (Cate …
    iFc : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι (F.m …
    ⊢ Nonempty (CategoryTheory.Limits.IsLimit (F.mapCone c))
  -/
  constructor
  /-
    case preserves.val
    C : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁵ : CategoryTheory.Preadditive C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    inst✝³ : CategoryTheory.Preadditive D
    F : CategoryTheory.Functor C D
    inst✝² : F.PreservesZeroMorphisms
    inst✝¹ : CategoryTheory.Limits.HasBinaryBiproducts C
    inst✝ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.Limits.PreservesLimit …
    X Y : C
    f g : Quiver.Hom X Y
    this✝ : CategoryTheory.Limits.PreservesBinaryBiproducts F := CategoryTheory.Li …
    this : F.Additive
    c : CategoryTheory.Limits.Cone (CategoryTheory.Limits.parallelPair f g)
    i : CategoryTheory.Limits.IsLimit c
    c' : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι (Cate …
    iFc : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι (F.m …
    ⊢ CategoryTheory.Limits.IsLimit (F.mapCone c)
  -/
  apply IsLimit.ofIsoLimit _ ((Cones.functoriality _ F).mapIso (Fork.isoForkOfι c).symm)
  /-
    C : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁵ : CategoryTheory.Preadditive C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    inst✝³ : CategoryTheory.Preadditive D
    F : CategoryTheory.Functor C D
    inst✝² : F.PreservesZeroMorphisms
    inst✝¹ : CategoryTheory.Limits.HasBinaryBiproducts C
    inst✝ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.Limits.PreservesLimit …
    X Y : C
    f g : Quiver.Hom X Y
    this✝ : CategoryTheory.Limits.PreservesBinaryBiproducts F := CategoryTheory.Li …
    this : F.Additive
    c : CategoryTheory.Limits.Cone (CategoryTheory.Limits.parallelPair f g)
    i : CategoryTheory.Limits.IsLimit c
    c' : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι (Cate …
    iFc : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι (F.m …
    ⊢ CategoryTheory.Limits.IsLimit ((CategoryTheory.Limits.Cones.functoriality (C …
  -/
  apply (isLimitMapConeForkEquiv F (Fork.condition c)).invFun
  let p : parallelPair (F.map (f - g)) 0 ≅ parallelPair (F.map f - F.map g) 0 :=
    parallelPair.eqOfHomEq F.map_sub rfl
  exact
    IsLimit.ofIsoLimit
      (isLimitForkOfKernelFork ((IsLimit.postcomposeHomEquiv p _).symm iFc))
      (Fork.ext (Iso.refl _) (by simp [p]))


/-- A functor between preadditive categories preserves all equalizers if it preserves all kernels.
-/
lemma preservesEqualizers_of_preservesKernels
    [∀ {X Y} (f : X ⟶ Y), PreservesLimit (parallelPair f 0) F] :
    PreservesLimitsOfShape WalkingParallelPair F where
  preservesLimit {K} := by
    letI := preservesEqualizer_of_preservesKernels F (K.map WalkingParallelPairHom.left)
        (K.map WalkingParallelPairHom.right)
    /-
      C : Type u₁
      inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
      inst✝⁵ : CategoryTheory.Preadditive C
      D : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
      inst✝³ : CategoryTheory.Preadditive D
      F : CategoryTheory.Functor C D
      inst✝² : F.PreservesZeroMorphisms
      inst✝¹ : CategoryTheory.Limits.HasBinaryBiproducts C
      inst✝ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.Limits.PreservesLimit …
      K : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair C
      this : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.parallelPai …
      ⊢ CategoryTheory.Limits.PreservesLimit K F
    -/
    apply preservesLimit_of_iso_diagram F (diagramIsoParallelPair K).symm
    /-
      🎉 no goals
    -/


/-- A functor between preadditive categories which preserves kernels preserves all finite limits.
-/
lemma preservesFiniteLimits_of_preservesKernels [HasFiniteProducts C] [HasEqualizers C]
    [HasZeroObject C] [HasZeroObject D] [∀ {X Y} (f : X ⟶ Y), PreservesLimit (parallelPair f 0) F] :
    PreservesFiniteLimits F := by
  /-
    C : Type u₁
    inst✝¹⁰ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁹ : CategoryTheory.Preadditive C
    D : Type u₂
    inst✝⁸ : CategoryTheory.Category.{v₂, u₂} D
    inst✝⁷ : CategoryTheory.Preadditive D
    F : CategoryTheory.Functor C D
    inst✝⁶ : F.PreservesZeroMorphisms
    inst✝⁵ : CategoryTheory.Limits.HasBinaryBiproducts C
    inst✝⁴ : CategoryTheory.Limits.HasFiniteProducts C
    inst✝³ : CategoryTheory.Limits.HasEqualizers C
    inst✝² : CategoryTheory.Limits.HasZeroObject C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject D
    inst✝ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.Limits.PreservesLimit …
    ⊢ CategoryTheory.Limits.PreservesFiniteLimits F
  -/
  letI := preservesEqualizers_of_preservesKernels F
  /-
    C : Type u₁
    inst✝¹⁰ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁹ : CategoryTheory.Preadditive C
    D : Type u₂
    inst✝⁸ : CategoryTheory.Category.{v₂, u₂} D
    inst✝⁷ : CategoryTheory.Preadditive D
    F : CategoryTheory.Functor C D
    inst✝⁶ : F.PreservesZeroMorphisms
    inst✝⁵ : CategoryTheory.Limits.HasBinaryBiproducts C
    inst✝⁴ : CategoryTheory.Limits.HasFiniteProducts C
    inst✝³ : CategoryTheory.Limits.HasEqualizers C
    inst✝² : CategoryTheory.Limits.HasZeroObject C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject D
    inst✝ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.Limits.PreservesLimit …
    this : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Walk …
    ⊢ CategoryTheory.Limits.PreservesFiniteLimits F
  -/
  letI := preservesTerminalObject_of_preservesZeroMorphisms F
  /-
    C : Type u₁
    inst✝¹⁰ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁹ : CategoryTheory.Preadditive C
    D : Type u₂
    inst✝⁸ : CategoryTheory.Category.{v₂, u₂} D
    inst✝⁷ : CategoryTheory.Preadditive D
    F : CategoryTheory.Functor C D
    inst✝⁶ : F.PreservesZeroMorphisms
    inst✝⁵ : CategoryTheory.Limits.HasBinaryBiproducts C
    inst✝⁴ : CategoryTheory.Limits.HasFiniteProducts C
    inst✝³ : CategoryTheory.Limits.HasEqualizers C
    inst✝² : CategoryTheory.Limits.HasZeroObject C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject D
    inst✝ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.Limits.PreservesLimit …
    this✝ : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wal …
    this : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Functor.empty C) F …
    ⊢ CategoryTheory.Limits.PreservesFiniteLimits F
  -/
  letI := preservesLimitsOfShape_pempty_of_preservesTerminal F
  /-
    C : Type u₁
    inst✝¹⁰ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁹ : CategoryTheory.Preadditive C
    D : Type u₂
    inst✝⁸ : CategoryTheory.Category.{v₂, u₂} D
    inst✝⁷ : CategoryTheory.Preadditive D
    F : CategoryTheory.Functor C D
    inst✝⁶ : F.PreservesZeroMorphisms
    inst✝⁵ : CategoryTheory.Limits.HasBinaryBiproducts C
    inst✝⁴ : CategoryTheory.Limits.HasFiniteProducts C
    inst✝³ : CategoryTheory.Limits.HasEqualizers C
    inst✝² : CategoryTheory.Limits.HasZeroObject C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject D
    inst✝ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.Limits.PreservesLimit …
    this✝¹ : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wa …
    this✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Functor.empty C)  …
    this : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete P …
    ⊢ CategoryTheory.Limits.PreservesFiniteLimits F
  -/
  letI : PreservesFiniteProducts F := ⟨preservesFiniteProducts_of_preserves_binary_and_terminal F⟩
  /-
    C : Type u₁
    inst✝¹⁰ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁹ : CategoryTheory.Preadditive C
    D : Type u₂
    inst✝⁸ : CategoryTheory.Category.{v₂, u₂} D
    inst✝⁷ : CategoryTheory.Preadditive D
    F : CategoryTheory.Functor C D
    inst✝⁶ : F.PreservesZeroMorphisms
    inst✝⁵ : CategoryTheory.Limits.HasBinaryBiproducts C
    inst✝⁴ : CategoryTheory.Limits.HasFiniteProducts C
    inst✝³ : CategoryTheory.Limits.HasEqualizers C
    inst✝² : CategoryTheory.Limits.HasZeroObject C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject D
    inst✝ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.Limits.PreservesLimit …
    this✝² : CategoryTheory.Limits.PreservesLimitsOfShape CategoryTheory.Limits.Wa …
    this✝¹ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Functor.empty C) …
    this✝ : CategoryTheory.Limits.PreservesLimitsOfShape (CategoryTheory.Discrete  …
    this : CategoryTheory.Limits.PreservesFiniteProducts F := { preserves := Categ …
    ⊢ CategoryTheory.Limits.PreservesFiniteLimits F
  -/
  exact preservesFiniteLimits_of_preservesEqualizers_and_finiteProducts F
  /-
    🎉 no goals
  -/


/-- A functor between preadditive categories which preserves cokernels preserves finite coproducts.
-/
def isColimitMapCoconeBinaryCofanOfPreservesCokernels {X Y Z : C} (ι₁ : X ⟶ Z) (ι₂ : Y ⟶ Z)
    [PreservesColimit (parallelPair ι₂ 0) F] (i : IsColimit (BinaryCofan.mk ι₁ ι₂)) :
    IsColimit (F.mapCocone (BinaryCofan.mk ι₁ ι₂)) := by
  /-
    C : Type u₁
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁴ : CategoryTheory.Preadditive C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    inst✝² : CategoryTheory.Preadditive D
    F : CategoryTheory.Functor C D
    inst✝¹ : F.PreservesZeroMorphisms
    X Y Z : C
    ι₁ : Quiver.Hom X Z
    ι₂ : Quiver.Hom Y Z
    inst✝ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.parallel …
    i : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk ι₁ ι₂)
    ⊢ CategoryTheory.Limits.IsColimit (F.mapCocone (CategoryTheory.Limits.BinaryCo …
  -/
  let bc := BinaryBicone.ofColimitCocone i
  /-
    C : Type u₁
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁴ : CategoryTheory.Preadditive C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    inst✝² : CategoryTheory.Preadditive D
    F : CategoryTheory.Functor C D
    inst✝¹ : F.PreservesZeroMorphisms
    X Y Z : C
    ι₁ : Quiver.Hom X Z
    ι₂ : Quiver.Hom Y Z
    inst✝ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.parallel …
    i : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk ι₁ ι₂)
    bc : CategoryTheory.Limits.BinaryBicone X Y := CategoryTheory.Limits.BinaryBic …
    ⊢ CategoryTheory.Limits.IsColimit (F.mapCocone (CategoryTheory.Limits.BinaryCo …
  -/
  let presf : PreservesColimit (parallelPair bc.inr 0) F := by simpa
  /-
    C : Type u₁
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁴ : CategoryTheory.Preadditive C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    inst✝² : CategoryTheory.Preadditive D
    F : CategoryTheory.Functor C D
    inst✝¹ : F.PreservesZeroMorphisms
    X Y Z : C
    ι₁ : Quiver.Hom X Z
    ι₂ : Quiver.Hom Y Z
    inst✝ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.parallel …
    i : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk ι₁ ι₂)
    bc : CategoryTheory.Limits.BinaryBicone X Y := CategoryTheory.Limits.BinaryBic …
    presf : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.parallel …
    ⊢ CategoryTheory.Limits.IsColimit (F.mapCocone (CategoryTheory.Limits.BinaryCo …
  -/
  let hf : IsColimit bc.inrCokernelCofork := BinaryBicone.isColimitInrCokernelCofork i
  exact
    (isColimitMapCoconeBinaryCofanEquiv F ι₁ ι₂).invFun
      (BinaryBicone.isBilimitOfCokernelFst (F.mapBinaryBicone bc)
          (isColimitMapCoconeCoforkEquiv' F bc.inr_fst (isColimitOfPreserves F hf))).isColimit


/-- A cokernel preserving functor between preadditive categories preserves any pair being
a colimit. -/
lemma preservesCoproduct_of_preservesCokernels
    [∀ {X Y} (f : X ⟶ Y), PreservesColimit (parallelPair f 0) F] {X Y : C} :
    PreservesColimit (pair X Y) F where
  preserves {c} hc :=
    ⟨IsColimit.ofIsoColimit
      (isColimitMapCoconeBinaryCofanOfPreservesCokernels F _ _
        (IsColimit.ofIsoColimit hc (isoBinaryCofanMk c)))
      ((Cocones.functoriality _ F).mapIso (isoBinaryCofanMk c).symm)⟩


/-- A cokernel preserving functor between preadditive categories preserves binary coproducts. -/
lemma preservesBinaryCoproducts_of_preservesCokernels
    [∀ {X Y} (f : X ⟶ Y), PreservesColimit (parallelPair f 0) F] :
    PreservesColimitsOfShape (Discrete WalkingPair) F where
  preservesColimit := preservesColimit_of_iso_diagram F (diagramIsoPair _).symm


/-- A functor between preadditive categories preserves the coequalizer of two
morphisms if it preserves all cokernels. -/
lemma preservesCoequalizer_of_preservesCokernels
    [∀ {X Y} (f : X ⟶ Y), PreservesColimit (parallelPair f 0) F] {X Y : C} (f g : X ⟶ Y) :
    PreservesColimit (parallelPair f g) F := by
  /-
    C : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁵ : CategoryTheory.Preadditive C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    inst✝³ : CategoryTheory.Preadditive D
    F : CategoryTheory.Functor C D
    inst✝² : F.PreservesZeroMorphisms
    inst✝¹ : CategoryTheory.Limits.HasBinaryBiproducts C
    inst✝ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.Limits.PreservesColim …
    X Y : C
    f g : Quiver.Hom X Y
    ⊢ CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.parallelPair f …
  -/
  letI := preservesBinaryBiproducts_of_preservesBinaryCoproducts F
  /-
    C : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁵ : CategoryTheory.Preadditive C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    inst✝³ : CategoryTheory.Preadditive D
    F : CategoryTheory.Functor C D
    inst✝² : F.PreservesZeroMorphisms
    inst✝¹ : CategoryTheory.Limits.HasBinaryBiproducts C
    inst✝ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.Limits.PreservesColim …
    X Y : C
    f g : Quiver.Hom X Y
    this : CategoryTheory.Limits.PreservesBinaryBiproducts F := CategoryTheory.Lim …
    ⊢ CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.parallelPair f …
  -/
  haveI := additive_of_preservesBinaryBiproducts F
  /-
    C : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁵ : CategoryTheory.Preadditive C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    inst✝³ : CategoryTheory.Preadditive D
    F : CategoryTheory.Functor C D
    inst✝² : F.PreservesZeroMorphisms
    inst✝¹ : CategoryTheory.Limits.HasBinaryBiproducts C
    inst✝ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.Limits.PreservesColim …
    X Y : C
    f g : Quiver.Hom X Y
    this✝ : CategoryTheory.Limits.PreservesBinaryBiproducts F := CategoryTheory.Li …
    this : F.Additive
    ⊢ CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.parallelPair f …
  -/
  constructor
  /-
    case preserves
    C : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁵ : CategoryTheory.Preadditive C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    inst✝³ : CategoryTheory.Preadditive D
    F : CategoryTheory.Functor C D
    inst✝² : F.PreservesZeroMorphisms
    inst✝¹ : CategoryTheory.Limits.HasBinaryBiproducts C
    inst✝ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.Limits.PreservesColim …
    X Y : C
    f g : Quiver.Hom X Y
    this✝ : CategoryTheory.Limits.PreservesBinaryBiproducts F := CategoryTheory.Li …
    this : F.Additive
    ⊢ ∀ {c : CategoryTheory.Limits.Cocone (CategoryTheory.Limits.parallelPair f g) …
  -/
  intro c i
  /-
    case preserves
    C : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁵ : CategoryTheory.Preadditive C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    inst✝³ : CategoryTheory.Preadditive D
    F : CategoryTheory.Functor C D
    inst✝² : F.PreservesZeroMorphisms
    inst✝¹ : CategoryTheory.Limits.HasBinaryBiproducts C
    inst✝ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.Limits.PreservesColim …
    X Y : C
    f g : Quiver.Hom X Y
    this✝ : CategoryTheory.Limits.PreservesBinaryBiproducts F := CategoryTheory.Li …
    this : F.Additive
    c : CategoryTheory.Limits.Cocone (CategoryTheory.Limits.parallelPair f g)
    i : CategoryTheory.Limits.IsColimit c
    ⊢ Nonempty (CategoryTheory.Limits.IsColimit (F.mapCocone c))
  -/
  let c' := isColimitCokernelCoforkOfCofork (i.ofIsoColimit (Cofork.isoCoforkOfπ c))
  /-
    case preserves
    C : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁵ : CategoryTheory.Preadditive C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    inst✝³ : CategoryTheory.Preadditive D
    F : CategoryTheory.Functor C D
    inst✝² : F.PreservesZeroMorphisms
    inst✝¹ : CategoryTheory.Limits.HasBinaryBiproducts C
    inst✝ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.Limits.PreservesColim …
    X Y : C
    f g : Quiver.Hom X Y
    this✝ : CategoryTheory.Limits.PreservesBinaryBiproducts F := CategoryTheory.Li …
    this : F.Additive
    c : CategoryTheory.Limits.Cocone (CategoryTheory.Limits.parallelPair f g)
    i : CategoryTheory.Limits.IsColimit c
    c' : CategoryTheory.Limits.IsColimit (CategoryTheory.Preadditive.cokernelCofor …
    ⊢ Nonempty (CategoryTheory.Limits.IsColimit (F.mapCocone c))
  -/
  dsimp only [cokernelCoforkOfCofork_ofπ] at c'
  /-
    case preserves
    C : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁵ : CategoryTheory.Preadditive C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    inst✝³ : CategoryTheory.Preadditive D
    F : CategoryTheory.Functor C D
    inst✝² : F.PreservesZeroMorphisms
    inst✝¹ : CategoryTheory.Limits.HasBinaryBiproducts C
    inst✝ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.Limits.PreservesColim …
    X Y : C
    f g : Quiver.Hom X Y
    this✝ : CategoryTheory.Limits.PreservesBinaryBiproducts F := CategoryTheory.Li …
    this : F.Additive
    c : CategoryTheory.Limits.Cocone (CategoryTheory.Limits.parallelPair f g)
    i : CategoryTheory.Limits.IsColimit c
    c' : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ …
    ⊢ Nonempty (CategoryTheory.Limits.IsColimit (F.mapCocone c))
  -/
  let iFc := isColimitCoforkMapOfIsColimit' F _ c'
  /-
    case preserves
    C : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁵ : CategoryTheory.Preadditive C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    inst✝³ : CategoryTheory.Preadditive D
    F : CategoryTheory.Functor C D
    inst✝² : F.PreservesZeroMorphisms
    inst✝¹ : CategoryTheory.Limits.HasBinaryBiproducts C
    inst✝ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.Limits.PreservesColim …
    X Y : C
    f g : Quiver.Hom X Y
    this✝ : CategoryTheory.Limits.PreservesBinaryBiproducts F := CategoryTheory.Li …
    this : F.Additive
    c : CategoryTheory.Limits.Cocone (CategoryTheory.Limits.parallelPair f g)
    i : CategoryTheory.Limits.IsColimit c
    c' : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ …
    iFc : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.of …
    ⊢ Nonempty (CategoryTheory.Limits.IsColimit (F.mapCocone c))
  -/
  constructor
  apply
    IsColimit.ofIsoColimit _ ((Cocones.functoriality _ F).mapIso (Cofork.isoCoforkOfπ c).symm)
  /-
    C : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁵ : CategoryTheory.Preadditive C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    inst✝³ : CategoryTheory.Preadditive D
    F : CategoryTheory.Functor C D
    inst✝² : F.PreservesZeroMorphisms
    inst✝¹ : CategoryTheory.Limits.HasBinaryBiproducts C
    inst✝ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.Limits.PreservesColim …
    X Y : C
    f g : Quiver.Hom X Y
    this✝ : CategoryTheory.Limits.PreservesBinaryBiproducts F := CategoryTheory.Li …
    this : F.Additive
    c : CategoryTheory.Limits.Cocone (CategoryTheory.Limits.parallelPair f g)
    i : CategoryTheory.Limits.IsColimit c
    c' : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ …
    iFc : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.of …
    ⊢ CategoryTheory.Limits.IsColimit ((CategoryTheory.Limits.Cocones.functorialit …
  -/
  apply (isColimitMapCoconeCoforkEquiv F (Cofork.condition c)).invFun
  let p : parallelPair (F.map (f - g)) 0 ≅ parallelPair (F.map f - F.map g) 0 :=
    parallelPair.ext (Iso.refl _) (Iso.refl _) (by simp) (by simp)
  exact
    IsColimit.ofIsoColimit
      (isColimitCoforkOfCokernelCofork ((IsColimit.precomposeHomEquiv p.symm _).symm iFc))
      (Cofork.ext (Iso.refl _) (by simp [p]))


/-- A functor between preadditive categories preserves all coequalizers if it preserves all kernels.
-/
lemma preservesCoequalizers_of_preservesCokernels
    [∀ {X Y} (f : X ⟶ Y), PreservesColimit (parallelPair f 0) F] :
    PreservesColimitsOfShape WalkingParallelPair F where
  preservesColimit {K} := by
    letI := preservesCoequalizer_of_preservesCokernels F (K.map Limits.WalkingParallelPairHom.left)
        (K.map Limits.WalkingParallelPairHom.right)
    /-
      C : Type u₁
      inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
      inst✝⁵ : CategoryTheory.Preadditive C
      D : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
      inst✝³ : CategoryTheory.Preadditive D
      F : CategoryTheory.Functor C D
      inst✝² : F.PreservesZeroMorphisms
      inst✝¹ : CategoryTheory.Limits.HasBinaryBiproducts C
      inst✝ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.Limits.PreservesColim …
      K : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair C
      this : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.parallelP …
      ⊢ CategoryTheory.Limits.PreservesColimit K F
    -/
    apply preservesColimit_of_iso_diagram F (diagramIsoParallelPair K).symm
    /-
      🎉 no goals
    -/


/-- A functor between preadditive categories which preserves kernels preserves all finite limits.
-/
lemma preservesFiniteColimits_of_preservesCokernels [HasFiniteCoproducts C] [HasCoequalizers C]
    [HasZeroObject C] [HasZeroObject D]
    [∀ {X Y} (f : X ⟶ Y), PreservesColimit (parallelPair f 0) F] : PreservesFiniteColimits F := by
  /-
    C : Type u₁
    inst✝¹⁰ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁹ : CategoryTheory.Preadditive C
    D : Type u₂
    inst✝⁸ : CategoryTheory.Category.{v₂, u₂} D
    inst✝⁷ : CategoryTheory.Preadditive D
    F : CategoryTheory.Functor C D
    inst✝⁶ : F.PreservesZeroMorphisms
    inst✝⁵ : CategoryTheory.Limits.HasBinaryBiproducts C
    inst✝⁴ : CategoryTheory.Limits.HasFiniteCoproducts C
    inst✝³ : CategoryTheory.Limits.HasCoequalizers C
    inst✝² : CategoryTheory.Limits.HasZeroObject C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject D
    inst✝ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.Limits.PreservesColim …
    ⊢ CategoryTheory.Limits.PreservesFiniteColimits F
  -/
  letI := preservesCoequalizers_of_preservesCokernels F
  /-
    C : Type u₁
    inst✝¹⁰ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁹ : CategoryTheory.Preadditive C
    D : Type u₂
    inst✝⁸ : CategoryTheory.Category.{v₂, u₂} D
    inst✝⁷ : CategoryTheory.Preadditive D
    F : CategoryTheory.Functor C D
    inst✝⁶ : F.PreservesZeroMorphisms
    inst✝⁵ : CategoryTheory.Limits.HasBinaryBiproducts C
    inst✝⁴ : CategoryTheory.Limits.HasFiniteCoproducts C
    inst✝³ : CategoryTheory.Limits.HasCoequalizers C
    inst✝² : CategoryTheory.Limits.HasZeroObject C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject D
    inst✝ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.Limits.PreservesColim …
    this : CategoryTheory.Limits.PreservesColimitsOfShape CategoryTheory.Limits.Wa …
    ⊢ CategoryTheory.Limits.PreservesFiniteColimits F
  -/
  letI := preservesInitialObject_of_preservesZeroMorphisms F
  /-
    C : Type u₁
    inst✝¹⁰ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁹ : CategoryTheory.Preadditive C
    D : Type u₂
    inst✝⁸ : CategoryTheory.Category.{v₂, u₂} D
    inst✝⁷ : CategoryTheory.Preadditive D
    F : CategoryTheory.Functor C D
    inst✝⁶ : F.PreservesZeroMorphisms
    inst✝⁵ : CategoryTheory.Limits.HasBinaryBiproducts C
    inst✝⁴ : CategoryTheory.Limits.HasFiniteCoproducts C
    inst✝³ : CategoryTheory.Limits.HasCoequalizers C
    inst✝² : CategoryTheory.Limits.HasZeroObject C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject D
    inst✝ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.Limits.PreservesColim …
    this✝ : CategoryTheory.Limits.PreservesColimitsOfShape CategoryTheory.Limits.W …
    this : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Functor.empty C) …
    ⊢ CategoryTheory.Limits.PreservesFiniteColimits F
  -/
  letI := preservesColimitsOfShape_pempty_of_preservesInitial F
  /-
    C : Type u₁
    inst✝¹⁰ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁹ : CategoryTheory.Preadditive C
    D : Type u₂
    inst✝⁸ : CategoryTheory.Category.{v₂, u₂} D
    inst✝⁷ : CategoryTheory.Preadditive D
    F : CategoryTheory.Functor C D
    inst✝⁶ : F.PreservesZeroMorphisms
    inst✝⁵ : CategoryTheory.Limits.HasBinaryBiproducts C
    inst✝⁴ : CategoryTheory.Limits.HasFiniteCoproducts C
    inst✝³ : CategoryTheory.Limits.HasCoequalizers C
    inst✝² : CategoryTheory.Limits.HasZeroObject C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject D
    inst✝ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.Limits.PreservesColim …
    this✝¹ : CategoryTheory.Limits.PreservesColimitsOfShape CategoryTheory.Limits. …
    this✝ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Functor.empty C …
    this : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discrete …
    ⊢ CategoryTheory.Limits.PreservesFiniteColimits F
  -/
  letI : PreservesFiniteCoproducts F := ⟨preservesFiniteCoproductsOfPreservesBinaryAndInitial F⟩
  /-
    C : Type u₁
    inst✝¹⁰ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁹ : CategoryTheory.Preadditive C
    D : Type u₂
    inst✝⁸ : CategoryTheory.Category.{v₂, u₂} D
    inst✝⁷ : CategoryTheory.Preadditive D
    F : CategoryTheory.Functor C D
    inst✝⁶ : F.PreservesZeroMorphisms
    inst✝⁵ : CategoryTheory.Limits.HasBinaryBiproducts C
    inst✝⁴ : CategoryTheory.Limits.HasFiniteCoproducts C
    inst✝³ : CategoryTheory.Limits.HasCoequalizers C
    inst✝² : CategoryTheory.Limits.HasZeroObject C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject D
    inst✝ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.Limits.PreservesColim …
    this✝² : CategoryTheory.Limits.PreservesColimitsOfShape CategoryTheory.Limits. …
    this✝¹ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Functor.empty  …
    this✝ : CategoryTheory.Limits.PreservesColimitsOfShape (CategoryTheory.Discret …
    this : CategoryTheory.Limits.PreservesFiniteCoproducts F := { preserves := Cat …
    ⊢ CategoryTheory.Limits.PreservesFiniteColimits F
  -/
  exact preservesFiniteColimits_of_preservesCoequalizers_and_finiteCoproducts F
  /-
    🎉 no goals
  -/


