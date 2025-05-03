/-- The `AffineTargetMorphismProperty` associated to `(targetAffineLocally P).diagonal`.
See `diagonal_targetAffineLocally_eq_targetAffineLocally`.
-/
def AffineTargetMorphismProperty.diagonal (P : AffineTargetMorphismProperty) :
    AffineTargetMorphismProperty :=
  fun {X _} f _ =>
    ∀ ⦃U₁ U₂ : Scheme⦄ (f₁ : U₁ ⟶ X) (f₂ : U₂ ⟶ X) [IsAffine U₁] [IsAffine U₂] [IsOpenImmersion f₁]
      [IsOpenImmersion f₂], P (pullback.mapDesc f₁ f₂ f)


instance AffineTargetMorphismProperty.diagonal_respectsIso (P : AffineTargetMorphismProperty)
    [P.toProperty.RespectsIso] : P.diagonal.toProperty.RespectsIso := by
  /-
    P : AlgebraicGeometry.AffineTargetMorphismProperty
    inst✝ : P.toProperty.RespectsIso
    ⊢ P.diagonal.toProperty.RespectsIso
  -/
  delta AffineTargetMorphismProperty.diagonal
  /-
    P : AlgebraicGeometry.AffineTargetMorphismProperty
    inst✝ : P.toProperty.RespectsIso
    ⊢ (AlgebraicGeometry.AffineTargetMorphismProperty.toProperty fun {X x} f x_1 = …
  -/
  apply AffineTargetMorphismProperty.respectsIso_mk
    /-
      case h₁
      P : AlgebraicGeometry.AffineTargetMorphismProperty
      inst✝ : P.toProperty.RespectsIso
      ⊢ ∀ {X Y Z : AlgebraicGeometry.Scheme} (e : CategoryTheory.Iso X Y) (f : Quive …
    -/
  · introv H _ _
    /-
      case h₁
      P : AlgebraicGeometry.AffineTargetMorphismProperty
      inst✝⁵ : P.toProperty.RespectsIso
      X Y Z : AlgebraicGeometry.Scheme
      e : CategoryTheory.Iso X Y
      f : Quiver.Hom Y Z
      inst✝⁴ : AlgebraicGeometry.IsAffine Z
      H : ∀ ⦃U₁ U₂ : AlgebraicGeometry.Scheme⦄ (f₁ : Quiver.Hom U₁ Y) (f₂ : Quiver.H …
      U₁ U₂ : AlgebraicGeometry.Scheme
      f₁ : Quiver.Hom U₁ X
      f₂ : Quiver.Hom U₂ X
      inst✝³ : AlgebraicGeometry.IsAffine U₁
      inst✝² : AlgebraicGeometry.IsAffine U₂
      inst✝¹ : AlgebraicGeometry.IsOpenImmersion f₁
      inst✝ : AlgebraicGeometry.IsOpenImmersion f₂
      ⊢ P (CategoryTheory.Limits.pullback.mapDesc f₁ f₂ (CategoryTheory.CategoryStru …
    -/
    rw [pullback.mapDesc_comp, P.cancel_left_of_respectsIso, P.cancel_right_of_respectsIso]
    /-
      case h₁
      P : AlgebraicGeometry.AffineTargetMorphismProperty
      inst✝⁵ : P.toProperty.RespectsIso
      X Y Z : AlgebraicGeometry.Scheme
      e : CategoryTheory.Iso X Y
      f : Quiver.Hom Y Z
      inst✝⁴ : AlgebraicGeometry.IsAffine Z
      H : ∀ ⦃U₁ U₂ : AlgebraicGeometry.Scheme⦄ (f₁ : Quiver.Hom U₁ Y) (f₂ : Quiver.H …
      U₁ U₂ : AlgebraicGeometry.Scheme
      f₁ : Quiver.Hom U₁ X
      f₂ : Quiver.Hom U₂ X
      inst✝³ : AlgebraicGeometry.IsAffine U₁
      inst✝² : AlgebraicGeometry.IsAffine U₂
      inst✝¹ : AlgebraicGeometry.IsOpenImmersion f₁
      inst✝ : AlgebraicGeometry.IsOpenImmersion f₂
      ⊢ P (CategoryTheory.Limits.pullback.mapDesc (CategoryTheory.CategoryStruct.com …
    -/
    apply H
    /-
      🎉 no goals
    -/
    /-
      case h₂
      P : AlgebraicGeometry.AffineTargetMorphismProperty
      inst✝ : P.toProperty.RespectsIso
      ⊢ ∀ {X Y Z : AlgebraicGeometry.Scheme} (e : CategoryTheory.Iso Y Z) (f : Quive …
    -/
  · introv H _ _
    /-
      case h₂
      P : AlgebraicGeometry.AffineTargetMorphismProperty
      inst✝⁵ : P.toProperty.RespectsIso
      X Y Z : AlgebraicGeometry.Scheme
      e : CategoryTheory.Iso Y Z
      f : Quiver.Hom X Y
      inst✝⁴ : AlgebraicGeometry.IsAffine Y
      H : ∀ ⦃U₁ U₂ : AlgebraicGeometry.Scheme⦄ (f₁ : Quiver.Hom U₁ X) (f₂ : Quiver.H …
      U₁ U₂ : AlgebraicGeometry.Scheme
      f₁ : Quiver.Hom U₁ X
      f₂ : Quiver.Hom U₂ X
      inst✝³ : AlgebraicGeometry.IsAffine U₁
      inst✝² : AlgebraicGeometry.IsAffine U₂
      inst✝¹ : AlgebraicGeometry.IsOpenImmersion f₁
      inst✝ : AlgebraicGeometry.IsOpenImmersion f₂
      ⊢ P (CategoryTheory.Limits.pullback.mapDesc f₁ f₂ (CategoryTheory.CategoryStru …
    -/
    rw [pullback.mapDesc_comp, P.cancel_right_of_respectsIso]
    /-
      case h₂
      P : AlgebraicGeometry.AffineTargetMorphismProperty
      inst✝⁵ : P.toProperty.RespectsIso
      X Y Z : AlgebraicGeometry.Scheme
      e : CategoryTheory.Iso Y Z
      f : Quiver.Hom X Y
      inst✝⁴ : AlgebraicGeometry.IsAffine Y
      H : ∀ ⦃U₁ U₂ : AlgebraicGeometry.Scheme⦄ (f₁ : Quiver.Hom U₁ X) (f₂ : Quiver.H …
      U₁ U₂ : AlgebraicGeometry.Scheme
      f₁ : Quiver.Hom U₁ X
      f₂ : Quiver.Hom U₂ X
      inst✝³ : AlgebraicGeometry.IsAffine U₁
      inst✝² : AlgebraicGeometry.IsAffine U₂
      inst✝¹ : AlgebraicGeometry.IsOpenImmersion f₁
      inst✝ : AlgebraicGeometry.IsOpenImmersion f₂
      ⊢ P (CategoryTheory.Limits.pullback.mapDesc f₁ f₂ f)
    -/
    apply H
    /-
      🎉 no goals
    -/


theorem HasAffineProperty.diagonal_of_openCover (P) {Q} [HasAffineProperty P Q]
    {X Y : Scheme.{u}} (f : X ⟶ Y) (𝒰 : Scheme.OpenCover.{u} Y) [∀ i, IsAffine (𝒰.obj i)]
    (𝒰' : ∀ i, Scheme.OpenCover.{u} (pullback f (𝒰.map i))) [∀ i j, IsAffine ((𝒰' i).obj j)]
    (h𝒰' : ∀ i j k,
      Q (pullback.mapDesc ((𝒰' i).map j) ((𝒰' i).map k) (𝒰.pullbackHom f i))) :
    P.diagonal f := by
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    Q : AlgebraicGeometry.AffineTargetMorphismProperty
    inst✝² : AlgebraicGeometry.HasAffineProperty P Q
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    𝒰 : Y.OpenCover
    inst✝¹ : ∀ (i : 𝒰.J), AlgebraicGeometry.IsAffine (𝒰.obj i)
    𝒰' : (i : 𝒰.J) → (CategoryTheory.Limits.pullback f (𝒰.map i)).OpenCover
    inst✝ : ∀ (i : 𝒰.J) (j : (𝒰' i).J), AlgebraicGeometry.IsAffine ((𝒰' i).obj j)
    h𝒰' : ∀ (i : 𝒰.J) (j k : (𝒰' i).J), Q (CategoryTheory.Limits.pullback.mapDesc  …
    ⊢ P.diagonal f
  -/
  letI := isLocal_affineProperty P
  let 𝒱 := (Scheme.Pullback.openCoverOfBase 𝒰 f f).bind fun i =>
    Scheme.Pullback.openCoverOfLeftRight.{u} (𝒰' i) (𝒰' i) (pullback.snd _ _) (pullback.snd _ _)
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    Q : AlgebraicGeometry.AffineTargetMorphismProperty
    inst✝² : AlgebraicGeometry.HasAffineProperty P Q
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    𝒰 : Y.OpenCover
    inst✝¹ : ∀ (i : 𝒰.J), AlgebraicGeometry.IsAffine (𝒰.obj i)
    𝒰' : (i : 𝒰.J) → (CategoryTheory.Limits.pullback f (𝒰.map i)).OpenCover
    inst✝ : ∀ (i : 𝒰.J) (j : (𝒰' i).J), AlgebraicGeometry.IsAffine ((𝒰' i).obj j)
    h𝒰' : ∀ (i : 𝒰.J) (j k : (𝒰' i).J), Q (CategoryTheory.Limits.pullback.mapDesc  …
    this : Q.IsLocal := AlgebraicGeometry.HasAffineProperty.isLocal_affineProperty P
    𝒱 : AlgebraicGeometry.Scheme.Cover (@AlgebraicGeometry.IsOpenImmersion) (Categ …
    ⊢ P.diagonal f
  -/
  have i1 : ∀ i, IsAffine (𝒱.obj i) := fun i => by dsimp [𝒱]; infer_instance
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    Q : AlgebraicGeometry.AffineTargetMorphismProperty
    inst✝² : AlgebraicGeometry.HasAffineProperty P Q
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    𝒰 : Y.OpenCover
    inst✝¹ : ∀ (i : 𝒰.J), AlgebraicGeometry.IsAffine (𝒰.obj i)
    𝒰' : (i : 𝒰.J) → (CategoryTheory.Limits.pullback f (𝒰.map i)).OpenCover
    inst✝ : ∀ (i : 𝒰.J) (j : (𝒰' i).J), AlgebraicGeometry.IsAffine ((𝒰' i).obj j)
    h𝒰' : ∀ (i : 𝒰.J) (j k : (𝒰' i).J), Q (CategoryTheory.Limits.pullback.mapDesc  …
    this : Q.IsLocal := AlgebraicGeometry.HasAffineProperty.isLocal_affineProperty P
    𝒱 : AlgebraicGeometry.Scheme.Cover (@AlgebraicGeometry.IsOpenImmersion) (Categ …
    i1 : ∀ (i : 𝒱.J), AlgebraicGeometry.IsAffine (𝒱.obj i)
    ⊢ P.diagonal f
  -/
  apply of_openCover 𝒱
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    Q : AlgebraicGeometry.AffineTargetMorphismProperty
    inst✝² : AlgebraicGeometry.HasAffineProperty P Q
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    𝒰 : Y.OpenCover
    inst✝¹ : ∀ (i : 𝒰.J), AlgebraicGeometry.IsAffine (𝒰.obj i)
    𝒰' : (i : 𝒰.J) → (CategoryTheory.Limits.pullback f (𝒰.map i)).OpenCover
    inst✝ : ∀ (i : 𝒰.J) (j : (𝒰' i).J), AlgebraicGeometry.IsAffine ((𝒰' i).obj j)
    h𝒰' : ∀ (i : 𝒰.J) (j k : (𝒰' i).J), Q (CategoryTheory.Limits.pullback.mapDesc  …
    this : Q.IsLocal := AlgebraicGeometry.HasAffineProperty.isLocal_affineProperty P
    𝒱 : AlgebraicGeometry.Scheme.Cover (@AlgebraicGeometry.IsOpenImmersion) (Categ …
    i1 : ∀ (i : 𝒱.J), AlgebraicGeometry.IsAffine (𝒱.obj i)
    ⊢ ∀ (i : 𝒱.1), Q (𝒱.pullbackHom (CategoryTheory.Limits.pullback.diagonal f) i)
  -/
  rintro ⟨i, j, k⟩
  /-
    case mk.mk
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    Q : AlgebraicGeometry.AffineTargetMorphismProperty
    inst✝² : AlgebraicGeometry.HasAffineProperty P Q
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    𝒰 : Y.OpenCover
    inst✝¹ : ∀ (i : 𝒰.J), AlgebraicGeometry.IsAffine (𝒰.obj i)
    𝒰' : (i : 𝒰.J) → (CategoryTheory.Limits.pullback f (𝒰.map i)).OpenCover
    inst✝ : ∀ (i : 𝒰.J) (j : (𝒰' i).J), AlgebraicGeometry.IsAffine ((𝒰' i).obj j)
    h𝒰' : ∀ (i : 𝒰.J) (j k : (𝒰' i).J), Q (CategoryTheory.Limits.pullback.mapDesc  …
    this : Q.IsLocal := AlgebraicGeometry.HasAffineProperty.isLocal_affineProperty P
    𝒱 : AlgebraicGeometry.Scheme.Cover (@AlgebraicGeometry.IsOpenImmersion) (Categ …
    i1 : ∀ (i : 𝒱.J), AlgebraicGeometry.IsAffine (𝒱.obj i)
    i : (AlgebraicGeometry.Scheme.Pullback.openCoverOfBase 𝒰 f f).J
    j k : (𝒰' i).J
    ⊢ Q (𝒱.pullbackHom (CategoryTheory.Limits.pullback.diagonal f) ⟨i, { fst := j, …
  -/
  dsimp [𝒱]
  convert (Q.cancel_left_of_respectsIso
    ((pullbackDiagonalMapIso _ _ ((𝒰' i).map j) ((𝒰' i).map k)).inv ≫
      pullback.map _ _ _ _ (𝟙 _) (𝟙 _) (𝟙 _) _ _) (pullback.snd _ _)).mp _ using 1
    /-
      case mk.mk.convert_1
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      Q : AlgebraicGeometry.AffineTargetMorphismProperty
      inst✝² : AlgebraicGeometry.HasAffineProperty P Q
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      𝒰 : Y.OpenCover
      inst✝¹ : ∀ (i : 𝒰.J), AlgebraicGeometry.IsAffine (𝒰.obj i)
      𝒰' : (i : 𝒰.J) → (CategoryTheory.Limits.pullback f (𝒰.map i)).OpenCover
      inst✝ : ∀ (i : 𝒰.J) (j : (𝒰' i).J), AlgebraicGeometry.IsAffine ((𝒰' i).obj j)
      h𝒰' : ∀ (i : 𝒰.J) (j k : (𝒰' i).J), Q (CategoryTheory.Limits.pullback.mapDesc  …
      this : Q.IsLocal := AlgebraicGeometry.HasAffineProperty.isLocal_affineProperty P
      𝒱 : AlgebraicGeometry.Scheme.Cover (@AlgebraicGeometry.IsOpenImmersion) (Categ …
      i1 : ∀ (i : 𝒱.J), AlgebraicGeometry.IsAffine (𝒱.obj i)
      i : (AlgebraicGeometry.Scheme.Pullback.openCoverOfBase 𝒰 f f).J
      j k : (𝒰' i).J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.diago …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case mk.mk.convert_2
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      Q : AlgebraicGeometry.AffineTargetMorphismProperty
      inst✝² : AlgebraicGeometry.HasAffineProperty P Q
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      𝒰 : Y.OpenCover
      inst✝¹ : ∀ (i : 𝒰.J), AlgebraicGeometry.IsAffine (𝒰.obj i)
      𝒰' : (i : 𝒰.J) → (CategoryTheory.Limits.pullback f (𝒰.map i)).OpenCover
      inst✝ : ∀ (i : 𝒰.J) (j : (𝒰' i).J), AlgebraicGeometry.IsAffine ((𝒰' i).obj j)
      h𝒰' : ∀ (i : 𝒰.J) (j k : (𝒰' i).J), Q (CategoryTheory.Limits.pullback.mapDesc  …
      this : Q.IsLocal := AlgebraicGeometry.HasAffineProperty.isLocal_affineProperty P
      𝒱 : AlgebraicGeometry.Scheme.Cover (@AlgebraicGeometry.IsOpenImmersion) (Categ …
      i1 : ∀ (i : 𝒱.J), AlgebraicGeometry.IsAffine (𝒱.obj i)
      i : (AlgebraicGeometry.Scheme.Pullback.openCoverOfBase 𝒰 f f).J
      j k : (𝒰' i).J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.map ( …
    -/
             /-
               🎉 no goals
             -/
  · ext1 <;> simp
             /-
               🎉 no goals
             -/
  · simp only [Category.assoc, limit.lift_π, PullbackCone.mk_pt, PullbackCone.mk_π_app,
      Functor.const_obj_obj, cospan_one, cospan_left, cospan_right, Category.comp_id]
    /-
      case mk.mk.convert_6
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      Q : AlgebraicGeometry.AffineTargetMorphismProperty
      inst✝² : AlgebraicGeometry.HasAffineProperty P Q
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      𝒰 : Y.OpenCover
      inst✝¹ : ∀ (i : 𝒰.J), AlgebraicGeometry.IsAffine (𝒰.obj i)
      𝒰' : (i : 𝒰.J) → (CategoryTheory.Limits.pullback f (𝒰.map i)).OpenCover
      inst✝ : ∀ (i : 𝒰.J) (j : (𝒰' i).J), AlgebraicGeometry.IsAffine ((𝒰' i).obj j)
      h𝒰' : ∀ (i : 𝒰.J) (j k : (𝒰' i).J), Q (CategoryTheory.Limits.pullback.mapDesc  …
      this : Q.IsLocal := AlgebraicGeometry.HasAffineProperty.isLocal_affineProperty P
      𝒱 : AlgebraicGeometry.Scheme.Cover (@AlgebraicGeometry.IsOpenImmersion) (Categ …
      i1 : ∀ (i : 𝒱.J), AlgebraicGeometry.IsAffine (𝒱.obj i)
      i : (AlgebraicGeometry.Scheme.Pullback.openCoverOfBase 𝒰 f f).J
      j k : (𝒰' i).J
      ⊢ Q (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullbackDiagona …
    -/
    convert h𝒰' i j k
    /-
      case h.e'_3.h
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      Q : AlgebraicGeometry.AffineTargetMorphismProperty
      inst✝² : AlgebraicGeometry.HasAffineProperty P Q
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      𝒰 : Y.OpenCover
      inst✝¹ : ∀ (i : 𝒰.J), AlgebraicGeometry.IsAffine (𝒰.obj i)
      𝒰' : (i : 𝒰.J) → (CategoryTheory.Limits.pullback f (𝒰.map i)).OpenCover
      inst✝ : ∀ (i : 𝒰.J) (j : (𝒰' i).J), AlgebraicGeometry.IsAffine ((𝒰' i).obj j)
      h𝒰' : ∀ (i : 𝒰.J) (j k : (𝒰' i).J), Q (CategoryTheory.Limits.pullback.mapDesc  …
      this : Q.IsLocal := AlgebraicGeometry.HasAffineProperty.isLocal_affineProperty P
      𝒱 : AlgebraicGeometry.Scheme.Cover (@AlgebraicGeometry.IsOpenImmersion) (Categ …
      i1 : ∀ (i : 𝒱.J), AlgebraicGeometry.IsAffine (𝒱.obj i)
      i : (AlgebraicGeometry.Scheme.Pullback.openCoverOfBase 𝒰 f f).J
      j k : (𝒰' i).J
      e_2✝ : Eq (CategoryTheory.Limits.pullback (CategoryTheory.CategoryStruct.comp  …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullbackDiagon …
    -/
             /-
               🎉 no goals
             -/
    ext1 <;> simp [Scheme.Cover.pullbackHom]
             /-
               🎉 no goals
             -/


theorem HasAffineProperty.diagonal_of_openCover_diagonal
    (P) {Q} [HasAffineProperty P Q]
    {X Y : Scheme.{u}} (f : X ⟶ Y) (𝒰 : Scheme.OpenCover.{u} Y) [∀ i, IsAffine (𝒰.obj i)]
    (h𝒰 : ∀ i, Q.diagonal (𝒰.pullbackHom f i)) :
    P.diagonal f :=
  diagonal_of_openCover P f 𝒰 (fun _ ↦ Scheme.affineCover _)
    (fun _ _ _ ↦ h𝒰 _ _ _)


theorem HasAffineProperty.diagonal_of_diagonal_of_isPullback
    (P) {Q} [HasAffineProperty P Q]
    {X Y U V : Scheme.{u}} {f : X ⟶ Y} {g : U ⟶ Y}
    [IsAffine U] [IsOpenImmersion g]
    {iV : V ⟶ X} {f' : V ⟶ U} (h : IsPullback iV f' f g) (H : P.diagonal f) :
    Q.diagonal f' := by
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    Q : AlgebraicGeometry.AffineTargetMorphismProperty
    inst✝² : AlgebraicGeometry.HasAffineProperty P Q
    X Y U V : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom U Y
    inst✝¹ : AlgebraicGeometry.IsAffine U
    inst✝ : AlgebraicGeometry.IsOpenImmersion g
    iV : Quiver.Hom V X
    f' : Quiver.Hom V U
    h : CategoryTheory.IsPullback iV f' f g
    H : P.diagonal f
    ⊢ Q.diagonal f'
  -/
  letI := isLocal_affineProperty P
  rw [← Q.diagonal.cancel_left_of_respectsIso h.isoPullback.inv,
    h.isoPullback_inv_snd]
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    Q : AlgebraicGeometry.AffineTargetMorphismProperty
    inst✝² : AlgebraicGeometry.HasAffineProperty P Q
    X Y U V : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom U Y
    inst✝¹ : AlgebraicGeometry.IsAffine U
    inst✝ : AlgebraicGeometry.IsOpenImmersion g
    iV : Quiver.Hom V X
    f' : Quiver.Hom V U
    h : CategoryTheory.IsPullback iV f' f g
    H : P.diagonal f
    this : Q.IsLocal := AlgebraicGeometry.HasAffineProperty.isLocal_affineProperty P
    ⊢ Q.diagonal (CategoryTheory.Limits.pullback.snd f g)
  -/
  rintro U V f₁ f₂ hU hV hf₁ hf₂
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    Q : AlgebraicGeometry.AffineTargetMorphismProperty
    inst✝² : AlgebraicGeometry.HasAffineProperty P Q
    X Y U✝ V✝ : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom U✝ Y
    inst✝¹ : AlgebraicGeometry.IsAffine U✝
    inst✝ : AlgebraicGeometry.IsOpenImmersion g
    iV : Quiver.Hom V✝ X
    f' : Quiver.Hom V✝ U✝
    h : CategoryTheory.IsPullback iV f' f g
    H : P.diagonal f
    this : Q.IsLocal := AlgebraicGeometry.HasAffineProperty.isLocal_affineProperty P
    U V : AlgebraicGeometry.Scheme
    f₁ : Quiver.Hom U (CategoryTheory.Limits.pullback f g)
    f₂ : Quiver.Hom V (CategoryTheory.Limits.pullback f g)
    hU : AlgebraicGeometry.IsAffine U
    hV : AlgebraicGeometry.IsAffine V
    hf₁ : AlgebraicGeometry.IsOpenImmersion f₁
    hf₂ : AlgebraicGeometry.IsOpenImmersion f₂
    ⊢ Q (CategoryTheory.Limits.pullback.mapDesc f₁ f₂ (CategoryTheory.Limits.pullb …
  -/
  rw [← Q.cancel_left_of_respectsIso (pullbackDiagonalMapIso f _ f₁ f₂).hom]
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    Q : AlgebraicGeometry.AffineTargetMorphismProperty
    inst✝² : AlgebraicGeometry.HasAffineProperty P Q
    X Y U✝ V✝ : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom U✝ Y
    inst✝¹ : AlgebraicGeometry.IsAffine U✝
    inst✝ : AlgebraicGeometry.IsOpenImmersion g
    iV : Quiver.Hom V✝ X
    f' : Quiver.Hom V✝ U✝
    h : CategoryTheory.IsPullback iV f' f g
    H : P.diagonal f
    this : Q.IsLocal := AlgebraicGeometry.HasAffineProperty.isLocal_affineProperty P
    U V : AlgebraicGeometry.Scheme
    f₁ : Quiver.Hom U (CategoryTheory.Limits.pullback f g)
    f₂ : Quiver.Hom V (CategoryTheory.Limits.pullback f g)
    hU : AlgebraicGeometry.IsAffine U
    hV : AlgebraicGeometry.IsAffine V
    hf₁ : AlgebraicGeometry.IsOpenImmersion f₁
    hf₂ : AlgebraicGeometry.IsOpenImmersion f₂
    ⊢ Q (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullbackDiagona …
  -/
  convert HasAffineProperty.of_isPullback (P := P) (.of_hasPullback _ _) H
    /-
      case h.e'_3.h
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      Q : AlgebraicGeometry.AffineTargetMorphismProperty
      inst✝² : AlgebraicGeometry.HasAffineProperty P Q
      X Y U✝ V✝ : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      g : Quiver.Hom U✝ Y
      inst✝¹ : AlgebraicGeometry.IsAffine U✝
      inst✝ : AlgebraicGeometry.IsOpenImmersion g
      iV : Quiver.Hom V✝ X
      f' : Quiver.Hom V✝ U✝
      h : CategoryTheory.IsPullback iV f' f g
      H : P.diagonal f
      this : Q.IsLocal := AlgebraicGeometry.HasAffineProperty.isLocal_affineProperty P
      U V : AlgebraicGeometry.Scheme
      f₁ : Quiver.Hom U (CategoryTheory.Limits.pullback f g)
      f₂ : Quiver.Hom V (CategoryTheory.Limits.pullback f g)
      hU : AlgebraicGeometry.IsAffine U
      hV : AlgebraicGeometry.IsAffine V
      hf₁ : AlgebraicGeometry.IsOpenImmersion f₁
      hf₂ : AlgebraicGeometry.IsOpenImmersion f₂
      e_1✝ : Eq (CategoryTheory.Limits.pullback (CategoryTheory.Limits.pullback.diag …
      e_2✝ : Eq (CategoryTheory.Limits.pullback (CategoryTheory.CategoryStruct.comp  …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullbackDiagon …
    -/
                               /-
                                 🎉 no goals
                               -/
  · apply pullback.hom_ext <;> simp
                               /-
                                 🎉 no goals
                               -/
    /-
      case convert_1
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      Q : AlgebraicGeometry.AffineTargetMorphismProperty
      inst✝² : AlgebraicGeometry.HasAffineProperty P Q
      X Y U✝ V✝ : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      g : Quiver.Hom U✝ Y
      inst✝¹ : AlgebraicGeometry.IsAffine U✝
      inst✝ : AlgebraicGeometry.IsOpenImmersion g
      iV : Quiver.Hom V✝ X
      f' : Quiver.Hom V✝ U✝
      h : CategoryTheory.IsPullback iV f' f g
      H : P.diagonal f
      this : Q.IsLocal := AlgebraicGeometry.HasAffineProperty.isLocal_affineProperty P
      U V : AlgebraicGeometry.Scheme
      f₁ : Quiver.Hom U (CategoryTheory.Limits.pullback f g)
      f₂ : Quiver.Hom V (CategoryTheory.Limits.pullback f g)
      hU : AlgebraicGeometry.IsAffine U
      hV : AlgebraicGeometry.IsAffine V
      hf₁ : AlgebraicGeometry.IsOpenImmersion f₁
      hf₂ : AlgebraicGeometry.IsOpenImmersion f₂
      ⊢ AlgebraicGeometry.IsAffine (CategoryTheory.Limits.pullback (CategoryTheory.C …
    -/
  · infer_instance
    /-
      🎉 no goals
    -/
    /-
      case convert_2
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      Q : AlgebraicGeometry.AffineTargetMorphismProperty
      inst✝² : AlgebraicGeometry.HasAffineProperty P Q
      X Y U✝ V✝ : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      g : Quiver.Hom U✝ Y
      inst✝¹ : AlgebraicGeometry.IsAffine U✝
      inst✝ : AlgebraicGeometry.IsOpenImmersion g
      iV : Quiver.Hom V✝ X
      f' : Quiver.Hom V✝ U✝
      h : CategoryTheory.IsPullback iV f' f g
      H : P.diagonal f
      this : Q.IsLocal := AlgebraicGeometry.HasAffineProperty.isLocal_affineProperty P
      U V : AlgebraicGeometry.Scheme
      f₁ : Quiver.Hom U (CategoryTheory.Limits.pullback f g)
      f₂ : Quiver.Hom V (CategoryTheory.Limits.pullback f g)
      hU : AlgebraicGeometry.IsAffine U
      hV : AlgebraicGeometry.IsAffine V
      hf₁ : AlgebraicGeometry.IsOpenImmersion f₁
      hf₂ : AlgebraicGeometry.IsOpenImmersion f₂
      ⊢ AlgebraicGeometry.IsOpenImmersion (CategoryTheory.Limits.pullback.map (Categ …
    -/
  · infer_instance
    /-
      🎉 no goals
    -/


theorem HasAffineProperty.diagonal_iff
    (P) {Q} [HasAffineProperty P Q] {X Y} {f : X ⟶ Y} [IsAffine Y] :
    Q.diagonal f ↔ P.diagonal f := by
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    Q : AlgebraicGeometry.AffineTargetMorphismProperty
    inst✝¹ : AlgebraicGeometry.HasAffineProperty P Q
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝ : AlgebraicGeometry.IsAffine Y
    ⊢ Iff (Q.diagonal f) (P.diagonal f)
  -/
  letI := isLocal_affineProperty P
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    Q : AlgebraicGeometry.AffineTargetMorphismProperty
    inst✝¹ : AlgebraicGeometry.HasAffineProperty P Q
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝ : AlgebraicGeometry.IsAffine Y
    this : Q.IsLocal := AlgebraicGeometry.HasAffineProperty.isLocal_affineProperty P
    ⊢ Iff (Q.diagonal f) (P.diagonal f)
  -/
  refine ⟨fun hf ↦ ?_, diagonal_of_diagonal_of_isPullback P .of_id_fst⟩
  rw [← Q.diagonal.cancel_left_of_respectsIso
    (pullback.fst (f := f) (g := 𝟙 Y)), pullback.condition, Category.comp_id] at hf
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    Q : AlgebraicGeometry.AffineTargetMorphismProperty
    inst✝¹ : AlgebraicGeometry.HasAffineProperty P Q
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝ : AlgebraicGeometry.IsAffine Y
    this : Q.IsLocal := AlgebraicGeometry.HasAffineProperty.isLocal_affineProperty P
    hf : Q.diagonal (CategoryTheory.Limits.pullback.snd f (CategoryTheory.Category …
    ⊢ P.diagonal f
  -/
  let 𝒰 := X.affineCover.pushforwardIso (inv (pullback.fst (f := f) (g := 𝟙 Y)))
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    Q : AlgebraicGeometry.AffineTargetMorphismProperty
    inst✝¹ : AlgebraicGeometry.HasAffineProperty P Q
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝ : AlgebraicGeometry.IsAffine Y
    this : Q.IsLocal := AlgebraicGeometry.HasAffineProperty.isLocal_affineProperty P
    hf : Q.diagonal (CategoryTheory.Limits.pullback.snd f (CategoryTheory.Category …
    𝒰 : AlgebraicGeometry.Scheme.Cover (@AlgebraicGeometry.IsOpenImmersion) (Categ …
    ⊢ P.diagonal f
  -/
  have (i) : IsAffine (𝒰.obj i) := by dsimp [𝒰]; infer_instance
  exact HasAffineProperty.diagonal_of_openCover P f (Scheme.coverOfIsIso (𝟙 _))
    (fun _ ↦ 𝒰) (fun _ _ _ ↦ hf _ _)


instance HasAffineProperty.diagonal_affineProperty_isLocal
    {Q : AffineTargetMorphismProperty} [Q.IsLocal] :
    Q.diagonal.IsLocal where
  respectsIso := inferInstance
  to_basicOpen {_ Y} _ f r hf :=
    diagonal_of_diagonal_of_isPullback (targetAffineLocally Q)
      (isPullback_morphismRestrict f (Y.basicOpen r)).flip
      ((diagonal_iff (targetAffineLocally Q)).mp hf)
  of_basicOpenCover {X Y} _ f s hs hs' := by
    /-
      Q : AlgebraicGeometry.AffineTargetMorphismProperty
      inst✝ : Q.IsLocal
      X Y : AlgebraicGeometry.Scheme
      x✝ : AlgebraicGeometry.IsAffine Y
      f : Quiver.Hom X Y
      s : Finset ↑(Y.presheaf.obj { unop := Top.top })
      hs : Eq (Ideal.span ↑s) Top.top
      hs' : ∀ (r : Subtype fun x => Membership.mem s x), Q.diagonal (AlgebraicGeomet …
      ⊢ Q.diagonal f
    -/
    refine (diagonal_iff (targetAffineLocally Q)).mpr ?_
    /-
      Q : AlgebraicGeometry.AffineTargetMorphismProperty
      inst✝ : Q.IsLocal
      X Y : AlgebraicGeometry.Scheme
      x✝ : AlgebraicGeometry.IsAffine Y
      f : Quiver.Hom X Y
      s : Finset ↑(Y.presheaf.obj { unop := Top.top })
      hs : Eq (Ideal.span ↑s) Top.top
      hs' : ∀ (r : Subtype fun x => Membership.mem s x), Q.diagonal (AlgebraicGeomet …
      ⊢ (AlgebraicGeometry.targetAffineLocally Q).diagonal f
    -/
    let 𝒰 := Y.openCoverOfISupEqTop _ (((isAffineOpen_top Y).basicOpen_union_eq_self_iff _).mpr hs)
    /-
      Q : AlgebraicGeometry.AffineTargetMorphismProperty
      inst✝ : Q.IsLocal
      X Y : AlgebraicGeometry.Scheme
      x✝ : AlgebraicGeometry.IsAffine Y
      f : Quiver.Hom X Y
      s : Finset ↑(Y.presheaf.obj { unop := Top.top })
      hs : Eq (Ideal.span ↑s) Top.top
      hs' : ∀ (r : Subtype fun x => Membership.mem s x), Q.diagonal (AlgebraicGeomet …
      𝒰 : Y.OpenCover := Y.openCoverOfISupEqTop (fun i => Y.basicOpen ↑i) ⋯
      ⊢ (AlgebraicGeometry.targetAffineLocally Q).diagonal f
    -/
    have (i) : IsAffine (𝒰.obj i) := (isAffineOpen_top Y).basicOpen i.1
    /-
      Q : AlgebraicGeometry.AffineTargetMorphismProperty
      inst✝ : Q.IsLocal
      X Y : AlgebraicGeometry.Scheme
      x✝ : AlgebraicGeometry.IsAffine Y
      f : Quiver.Hom X Y
      s : Finset ↑(Y.presheaf.obj { unop := Top.top })
      hs : Eq (Ideal.span ↑s) Top.top
      hs' : ∀ (r : Subtype fun x => Membership.mem s x), Q.diagonal (AlgebraicGeomet …
      𝒰 : Y.OpenCover := Y.openCoverOfISupEqTop (fun i => Y.basicOpen ↑i) ⋯
      this : ∀ (i : 𝒰.J), AlgebraicGeometry.IsAffine (𝒰.obj i)
      ⊢ (AlgebraicGeometry.targetAffineLocally Q).diagonal f
    -/
    refine diagonal_of_openCover_diagonal (targetAffineLocally Q) f 𝒰 ?_
    /-
      Q : AlgebraicGeometry.AffineTargetMorphismProperty
      inst✝ : Q.IsLocal
      X Y : AlgebraicGeometry.Scheme
      x✝ : AlgebraicGeometry.IsAffine Y
      f : Quiver.Hom X Y
      s : Finset ↑(Y.presheaf.obj { unop := Top.top })
      hs : Eq (Ideal.span ↑s) Top.top
      hs' : ∀ (r : Subtype fun x => Membership.mem s x), Q.diagonal (AlgebraicGeomet …
      𝒰 : Y.OpenCover := Y.openCoverOfISupEqTop (fun i => Y.basicOpen ↑i) ⋯
      this : ∀ (i : 𝒰.J), AlgebraicGeometry.IsAffine (𝒰.obj i)
      ⊢ ∀ (i : 𝒰.1), Q.diagonal (AlgebraicGeometry.Scheme.Cover.pullbackHom 𝒰 f i)
    -/
    intro i
    exact (Q.diagonal.arrow_mk_iso_iff
      (morphismRestrictEq _ (by simp [𝒰]) ≪≫ morphismRestrictOpensRange _ _)).mp (hs' i)


instance (P) {Q} [HasAffineProperty P Q] : HasAffineProperty P.diagonal Q.diagonal where
  isLocal_affineProperty := letI := HasAffineProperty.isLocal_affineProperty P; inferInstance
  eq_targetAffineLocally' := by
    /-
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      Q : AlgebraicGeometry.AffineTargetMorphismProperty
      inst✝ : AlgebraicGeometry.HasAffineProperty P Q
      ⊢ Eq P.diagonal (AlgebraicGeometry.targetAffineLocally Q.diagonal)
    -/
    ext X Y f
    /-
      case h
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      Q : AlgebraicGeometry.AffineTargetMorphismProperty
      inst✝ : AlgebraicGeometry.HasAffineProperty P Q
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      ⊢ Iff (P.diagonal f) (AlgebraicGeometry.targetAffineLocally Q.diagonal f)
    -/
    letI := HasAffineProperty.isLocal_affineProperty P
    /-
      case h
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      Q : AlgebraicGeometry.AffineTargetMorphismProperty
      inst✝ : AlgebraicGeometry.HasAffineProperty P Q
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      this : Q.IsLocal := AlgebraicGeometry.HasAffineProperty.isLocal_affineProperty P
      ⊢ Iff (P.diagonal f) (AlgebraicGeometry.targetAffineLocally Q.diagonal f)
    -/
    constructor
    · exact fun H U ↦ HasAffineProperty.diagonal_of_diagonal_of_isPullback P
        (isPullback_morphismRestrict f U).flip H
    · exact fun H ↦ HasAffineProperty.diagonal_of_openCover_diagonal P f Y.affineCover
        (fun i ↦ of_targetAffineLocally_of_isPullback (.of_hasPullback _ _) H)


instance (P) [IsLocalAtTarget P] : IsLocalAtTarget P.diagonal :=
  letI := HasAffineProperty.of_isLocalAtTarget P
  inferInstance


theorem universally_isLocalAtTarget (P : MorphismProperty Scheme)
    (hP₂ : ∀ {X Y : Scheme.{u}} (f : X ⟶ Y) {ι : Type u} (U : ι → Y.Opens)
      (_ : iSup U = ⊤), (∀ i, P (f ∣_ U i)) → P f) : IsLocalAtTarget P.universally := by
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    hP₂ : ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y) {ι : Type u} (U  …
    ⊢ AlgebraicGeometry.IsLocalAtTarget P.universally
  -/
  apply IsLocalAtTarget.mk'
  · exact fun {X Y} f U => P.universally.of_isPullback
      (isPullback_morphismRestrict f U).flip
    /-
      case of_sSup_eq_top
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      hP₂ : ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y) {ι : Type u} (U  …
      ⊢ ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y) {ι : Type u} (U : ι  …
    -/
  · intros X Y f ι U hU H X' Y' i₁ i₂ f' h
    /-
      case of_sSup_eq_top
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      hP₂ : ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y) {ι : Type u} (U  …
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      ι : Type u
      U : ι → Y.Opens
      hU : Eq (iSup U) Top.top
      H : ∀ (i : ι), P.universally (AlgebraicGeometry.morphismRestrict f (U i))
      X' Y' : AlgebraicGeometry.Scheme
      i₁ : Quiver.Hom X' X
      i₂ : Quiver.Hom Y' Y
      f' : Quiver.Hom X' Y'
      h : CategoryTheory.IsPullback f' i₁ i₂ f
      ⊢ P f'
    -/
    apply hP₂ _ (fun i ↦ i₂ ⁻¹ᵁ U i)
      /-
        case of_sSup_eq_top.x
        P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
        hP₂ : ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y) {ι : Type u} (U  …
        X Y : AlgebraicGeometry.Scheme
        f : Quiver.Hom X Y
        ι : Type u
        U : ι → Y.Opens
        hU : Eq (iSup U) Top.top
        H : ∀ (i : ι), P.universally (AlgebraicGeometry.morphismRestrict f (U i))
        X' Y' : AlgebraicGeometry.Scheme
        i₁ : Quiver.Hom X' X
        i₂ : Quiver.Hom Y' Y
        f' : Quiver.Hom X' Y'
        h : CategoryTheory.IsPullback f' i₁ i₂ f
        ⊢ Eq (iSup fun i => (TopologicalSpace.Opens.map i₂.base).obj (U i)) Top.top
      -/
    · rw [← top_le_iff] at hU ⊢
      /-
        case of_sSup_eq_top.x
        P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
        hP₂ : ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y) {ι : Type u} (U  …
        X Y : AlgebraicGeometry.Scheme
        f : Quiver.Hom X Y
        ι : Type u
        U : ι → Y.Opens
        hU : LE.le Top.top (iSup U)
        H : ∀ (i : ι), P.universally (AlgebraicGeometry.morphismRestrict f (U i))
        X' Y' : AlgebraicGeometry.Scheme
        i₁ : Quiver.Hom X' X
        i₂ : Quiver.Hom Y' Y
        f' : Quiver.Hom X' Y'
        h : CategoryTheory.IsPullback f' i₁ i₂ f
        ⊢ LE.le Top.top (iSup fun i => (TopologicalSpace.Opens.map i₂.base).obj (U i))
      -/
      rintro x -
      /-
        case of_sSup_eq_top.x
        P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
        hP₂ : ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y) {ι : Type u} (U  …
        X Y : AlgebraicGeometry.Scheme
        f : Quiver.Hom X Y
        ι : Type u
        U : ι → Y.Opens
        hU : LE.le Top.top (iSup U)
        H : ∀ (i : ι), P.universally (AlgebraicGeometry.morphismRestrict f (U i))
        X' Y' : AlgebraicGeometry.Scheme
        i₁ : Quiver.Hom X' X
        i₂ : Quiver.Hom Y' Y
        f' : Quiver.Hom X' Y'
        h : CategoryTheory.IsPullback f' i₁ i₂ f
        x : ↑↑Y'.toPresheafedSpace
        ⊢ Membership.mem (↑(iSup fun i => (TopologicalSpace.Opens.map i₂.base).obj (U  …
      -/
      simpa using @hU (i₂.base x) trivial
      /-
        🎉 no goals
      -/
      /-
        case of_sSup_eq_top.a
        P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
        hP₂ : ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y) {ι : Type u} (U  …
        X Y : AlgebraicGeometry.Scheme
        f : Quiver.Hom X Y
        ι : Type u
        U : ι → Y.Opens
        hU : Eq (iSup U) Top.top
        H : ∀ (i : ι), P.universally (AlgebraicGeometry.morphismRestrict f (U i))
        X' Y' : AlgebraicGeometry.Scheme
        i₁ : Quiver.Hom X' X
        i₂ : Quiver.Hom Y' Y
        f' : Quiver.Hom X' Y'
        h : CategoryTheory.IsPullback f' i₁ i₂ f
        ⊢ ∀ (i : ι), P (AlgebraicGeometry.morphismRestrict f' ((TopologicalSpace.Opens …
      -/
    · rintro i
      /-
        case of_sSup_eq_top.a
        P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
        hP₂ : ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y) {ι : Type u} (U  …
        X Y : AlgebraicGeometry.Scheme
        f : Quiver.Hom X Y
        ι : Type u
        U : ι → Y.Opens
        hU : Eq (iSup U) Top.top
        H : ∀ (i : ι), P.universally (AlgebraicGeometry.morphismRestrict f (U i))
        X' Y' : AlgebraicGeometry.Scheme
        i₁ : Quiver.Hom X' X
        i₂ : Quiver.Hom Y' Y
        f' : Quiver.Hom X' Y'
        h : CategoryTheory.IsPullback f' i₁ i₂ f
        i : ι
        ⊢ P (AlgebraicGeometry.morphismRestrict f' ((TopologicalSpace.Opens.map i₂.bas …
      -/
      refine H _ ((X'.isoOfEq ?_).hom ≫ i₁ ∣_ _) (i₂ ∣_ _) _ ?_
        /-
          case of_sSup_eq_top.a.refine_1
          P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
          hP₂ : ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y) {ι : Type u} (U  …
          X Y : AlgebraicGeometry.Scheme
          f : Quiver.Hom X Y
          ι : Type u
          U : ι → Y.Opens
          hU : Eq (iSup U) Top.top
          H : ∀ (i : ι), P.universally (AlgebraicGeometry.morphismRestrict f (U i))
          X' Y' : AlgebraicGeometry.Scheme
          i₁ : Quiver.Hom X' X
          i₂ : Quiver.Hom Y' Y
          f' : Quiver.Hom X' Y'
          h : CategoryTheory.IsPullback f' i₁ i₂ f
          i : ι
          ⊢ Eq ((TopologicalSpace.Opens.map f'.base).obj ((TopologicalSpace.Opens.map i₂ …
        -/
      · exact congr($(h.1.1) ⁻¹ᵁ U i)
        /-
          🎉 no goals
        -/
        /-
          case of_sSup_eq_top.a.refine_2
          P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
          hP₂ : ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y) {ι : Type u} (U  …
          X Y : AlgebraicGeometry.Scheme
          f : Quiver.Hom X Y
          ι : Type u
          U : ι → Y.Opens
          hU : Eq (iSup U) Top.top
          H : ∀ (i : ι), P.universally (AlgebraicGeometry.morphismRestrict f (U i))
          X' Y' : AlgebraicGeometry.Scheme
          i₁ : Quiver.Hom X' X
          i₂ : Quiver.Hom Y' Y
          f' : Quiver.Hom X' Y'
          h : CategoryTheory.IsPullback f' i₁ i₂ f
          i : ι
          ⊢ CategoryTheory.IsPullback (AlgebraicGeometry.morphismRestrict f' ((Topologic …
        -/
      · rw [← (isPullback_morphismRestrict f _).paste_vert_iff]
          /-
            case of_sSup_eq_top.a.refine_2
            P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
            hP₂ : ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y) {ι : Type u} (U  …
            X Y : AlgebraicGeometry.Scheme
            f : Quiver.Hom X Y
            ι : Type u
            U : ι → Y.Opens
            hU : Eq (iSup U) Top.top
            H : ∀ (i : ι), P.universally (AlgebraicGeometry.morphismRestrict f (U i))
            X' Y' : AlgebraicGeometry.Scheme
            i₁ : Quiver.Hom X' X
            i₂ : Quiver.Hom Y' Y
            f' : Quiver.Hom X' Y'
            h : CategoryTheory.IsPullback f' i₁ i₂ f
            i : ι
            ⊢ CategoryTheory.IsPullback (AlgebraicGeometry.morphismRestrict f' ((Topologic …
          -/
        · simp only [Category.assoc, morphismRestrict_ι, Scheme.isoOfEq_hom_ι_assoc]
          /-
            case of_sSup_eq_top.a.refine_2
            P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
            hP₂ : ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y) {ι : Type u} (U  …
            X Y : AlgebraicGeometry.Scheme
            f : Quiver.Hom X Y
            ι : Type u
            U : ι → Y.Opens
            hU : Eq (iSup U) Top.top
            H : ∀ (i : ι), P.universally (AlgebraicGeometry.morphismRestrict f (U i))
            X' Y' : AlgebraicGeometry.Scheme
            i₁ : Quiver.Hom X' X
            i₂ : Quiver.Hom Y' Y
            f' : Quiver.Hom X' Y'
            h : CategoryTheory.IsPullback f' i₁ i₂ f
            i : ι
            ⊢ CategoryTheory.IsPullback (AlgebraicGeometry.morphismRestrict f' ((Topologic …
          -/
          exact (isPullback_morphismRestrict f' (i₂ ⁻¹ᵁ U i)).paste_vert h
          /-
            🎉 no goals
          -/
          /-
            case of_sSup_eq_top.a.refine_2
            P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
            hP₂ : ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y) {ι : Type u} (U  …
            X Y : AlgebraicGeometry.Scheme
            f : Quiver.Hom X Y
            ι : Type u
            U : ι → Y.Opens
            hU : Eq (iSup U) Top.top
            H : ∀ (i : ι), P.universally (AlgebraicGeometry.morphismRestrict f (U i))
            X' Y' : AlgebraicGeometry.Scheme
            i₁ : Quiver.Hom X' X
            i₂ : Quiver.Hom Y' Y
            f' : Quiver.Hom X' Y'
            h : CategoryTheory.IsPullback f' i₁ i₂ f
            i : ι
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.morphismRestrict f …
          -/
        · rw [← cancel_mono (Scheme.Opens.ι _)]
          /-
            case of_sSup_eq_top.a.refine_2
            P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
            hP₂ : ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y) {ι : Type u} (U  …
            X Y : AlgebraicGeometry.Scheme
            f : Quiver.Hom X Y
            ι : Type u
            U : ι → Y.Opens
            hU : Eq (iSup U) Top.top
            H : ∀ (i : ι), P.universally (AlgebraicGeometry.morphismRestrict f (U i))
            X' Y' : AlgebraicGeometry.Scheme
            i₁ : Quiver.Hom X' X
            i₂ : Quiver.Hom Y' Y
            f' : Quiver.Hom X' Y'
            h : CategoryTheory.IsPullback f' i₁ i₂ f
            i : ι
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
          -/
          simp [morphismRestrict_ι_assoc, h.1.1]
          /-
            🎉 no goals
          -/


/-- `topologically P` holds for a morphism if the underlying topological map satisfies `P`. -/
def topologically
    (P : ∀ {α β : Type u} [TopologicalSpace α] [TopologicalSpace β] (_ : α → β), Prop) :
    MorphismProperty Scheme.{u} := fun _ _ f => P f.base


/-- If a property of maps of topological spaces is stable under composition, the induced
morphism property of schemes is stable under composition. -/
lemma topologically_isStableUnderComposition
    (hP : ∀ {α β γ : Type u} [TopologicalSpace α] [TopologicalSpace β] [TopologicalSpace γ]
      (f : α → β) (g : β → γ) (_ : P f) (_ : P g), P (g ∘ f)) :
    (topologically P).IsStableUnderComposition where
  comp_mem {X Y Z} f g hf hg := by
    /-
      P : {α β : Type u} → [inst : TopologicalSpace α] → [inst : TopologicalSpace β] …
      hP : ∀ {α β γ : Type u} [inst : TopologicalSpace α] [inst_1 : TopologicalSpace …
      X Y Z : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      hf : AlgebraicGeometry.topologically (fun {α β} [TopologicalSpace α] [Topologi …
      hg : AlgebraicGeometry.topologically (fun {α β} [TopologicalSpace α] [Topologi …
      ⊢ AlgebraicGeometry.topologically (fun {α β} [TopologicalSpace α] [Topological …
    -/
    simp only [topologically, Scheme.comp_coeBase, TopCat.coe_comp]
    /-
      P : {α β : Type u} → [inst : TopologicalSpace α] → [inst : TopologicalSpace β] …
      hP : ∀ {α β γ : Type u} [inst : TopologicalSpace α] [inst_1 : TopologicalSpace …
      X Y Z : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      hf : AlgebraicGeometry.topologically (fun {α β} [TopologicalSpace α] [Topologi …
      hg : AlgebraicGeometry.topologically (fun {α β} [TopologicalSpace α] [Topologi …
      ⊢ P (Function.comp ⇑g.base ⇑f.base)
    -/
    exact hP _ _ hf hg
    /-
      🎉 no goals
    -/


/-- If a property of maps of topological spaces is satisfied by all homeomorphisms,
every isomorphism of schemes satisfies the induced property. -/
lemma topologically_iso_le
    (hP : ∀ {α β : Type u} [TopologicalSpace α] [TopologicalSpace β] (f : α ≃ₜ β), P f) :
    MorphismProperty.isomorphisms Scheme ≤ (topologically P) := by
  /-
    P : {α β : Type u} → [inst : TopologicalSpace α] → [inst : TopologicalSpace β] …
    hP : ∀ {α β : Type u} [inst : TopologicalSpace α] [inst_1 : TopologicalSpace β …
    ⊢ LE.le (CategoryTheory.MorphismProperty.isomorphisms AlgebraicGeometry.Scheme …
  -/
  intro X Y e (he : IsIso e)
  /-
    P : {α β : Type u} → [inst : TopologicalSpace α] → [inst : TopologicalSpace β] …
    hP : ∀ {α β : Type u} [inst : TopologicalSpace α] [inst_1 : TopologicalSpace β …
    X Y : AlgebraicGeometry.Scheme
    e : Quiver.Hom X Y
    he : CategoryTheory.IsIso e
    ⊢ AlgebraicGeometry.topologically (fun {α β} [TopologicalSpace α] [Topological …
  -/
  have : IsIso e := he
  /-
    P : {α β : Type u} → [inst : TopologicalSpace α] → [inst : TopologicalSpace β] …
    hP : ∀ {α β : Type u} [inst : TopologicalSpace α] [inst_1 : TopologicalSpace β …
    X Y : AlgebraicGeometry.Scheme
    e : Quiver.Hom X Y
    he this : CategoryTheory.IsIso e
    ⊢ AlgebraicGeometry.topologically (fun {α β} [TopologicalSpace α] [Topological …
  -/
  exact hP (TopCat.homeoOfIso (asIso e.base))
  /-
    🎉 no goals
  -/


/-- If a property of maps of topological spaces is satisfied by homeomorphisms and is stable
under composition, the induced property on schemes respects isomorphisms. -/
lemma topologically_respectsIso
    (hP₁ : ∀ {α β : Type u} [TopologicalSpace α] [TopologicalSpace β] (f : α ≃ₜ β), P f)
    (hP₂ : ∀ {α β γ : Type u} [TopologicalSpace α] [TopologicalSpace β] [TopologicalSpace γ]
      (f : α → β) (g : β → γ) (_ : P f) (_ : P g), P (g ∘ f)) :
      (topologically P).RespectsIso :=
  have : (topologically P).IsStableUnderComposition :=
    topologically_isStableUnderComposition P hP₂
  MorphismProperty.respectsIso_of_isStableUnderComposition (topologically_iso_le P hP₁)


/-- To check that a topologically defined morphism property is local at the target,
we may check the corresponding properties on topological spaces. -/
lemma topologically_isLocalAtTarget
    [(topologically P).RespectsIso]
    (hP₂ : ∀ {α β : Type u} [TopologicalSpace α] [TopologicalSpace β] (f : α → β) (s : Set β)
      (_ : Continuous f) (_ : IsOpen s), P f → P (s.restrictPreimage f))
    (hP₃ : ∀ {α β : Type u} [TopologicalSpace α] [TopologicalSpace β] (f : α → β) {ι : Type u}
      (U : ι → TopologicalSpace.Opens β) (_ : iSup U = ⊤) (_ : Continuous f),
      (∀ i, P ((U i).carrier.restrictPreimage f)) → P f) :
    IsLocalAtTarget (topologically P) := by
  /-
    P : {α β : Type u} → [inst : TopologicalSpace α] → [inst : TopologicalSpace β] …
    inst✝ : (AlgebraicGeometry.topologically fun {α β} [TopologicalSpace α] [Topol …
    hP₂ : ∀ {α β : Type u} [inst : TopologicalSpace α] [inst_1 : TopologicalSpace  …
    hP₃ : ∀ {α β : Type u} [inst : TopologicalSpace α] [inst_1 : TopologicalSpace  …
    ⊢ AlgebraicGeometry.IsLocalAtTarget (AlgebraicGeometry.topologically fun {α β} …
  -/
  apply IsLocalAtTarget.mk'
    /-
      case restrict
      P : {α β : Type u} → [inst : TopologicalSpace α] → [inst : TopologicalSpace β] …
      inst✝ : (AlgebraicGeometry.topologically fun {α β} [TopologicalSpace α] [Topol …
      hP₂ : ∀ {α β : Type u} [inst : TopologicalSpace α] [inst_1 : TopologicalSpace  …
      hP₃ : ∀ {α β : Type u} [inst : TopologicalSpace α] [inst_1 : TopologicalSpace  …
      ⊢ ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y) (U : Y.Opens), Algeb …
    -/
  · intro X Y f U hf
    /-
      case restrict
      P : {α β : Type u} → [inst : TopologicalSpace α] → [inst : TopologicalSpace β] …
      inst✝ : (AlgebraicGeometry.topologically fun {α β} [TopologicalSpace α] [Topol …
      hP₂ : ∀ {α β : Type u} [inst : TopologicalSpace α] [inst_1 : TopologicalSpace  …
      hP₃ : ∀ {α β : Type u} [inst : TopologicalSpace α] [inst_1 : TopologicalSpace  …
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      U : Y.Opens
      hf : AlgebraicGeometry.topologically (fun {α β} [TopologicalSpace α] [Topologi …
      ⊢ AlgebraicGeometry.topologically (fun {α β} [TopologicalSpace α] [Topological …
    -/
    simp_rw [topologically, morphismRestrict_base]
    /-
      case restrict
      P : {α β : Type u} → [inst : TopologicalSpace α] → [inst : TopologicalSpace β] …
      inst✝ : (AlgebraicGeometry.topologically fun {α β} [TopologicalSpace α] [Topol …
      hP₂ : ∀ {α β : Type u} [inst : TopologicalSpace α] [inst_1 : TopologicalSpace  …
      hP₃ : ∀ {α β : Type u} [inst : TopologicalSpace α] [inst_1 : TopologicalSpace  …
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      U : Y.Opens
      hf : AlgebraicGeometry.topologically (fun {α β} [TopologicalSpace α] [Topologi …
      ⊢ P (U.carrier.restrictPreimage ⇑f.base)
    -/
    exact hP₂ f.base U.carrier f.base.2 U.2 hf
    /-
      🎉 no goals
    -/
    /-
      case of_sSup_eq_top
      P : {α β : Type u} → [inst : TopologicalSpace α] → [inst : TopologicalSpace β] …
      inst✝ : (AlgebraicGeometry.topologically fun {α β} [TopologicalSpace α] [Topol …
      hP₂ : ∀ {α β : Type u} [inst : TopologicalSpace α] [inst_1 : TopologicalSpace  …
      hP₃ : ∀ {α β : Type u} [inst : TopologicalSpace α] [inst_1 : TopologicalSpace  …
      ⊢ ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y) {ι : Type u} (U : ι  …
    -/
  · intro X Y f ι U hU hf
    /-
      case of_sSup_eq_top
      P : {α β : Type u} → [inst : TopologicalSpace α] → [inst : TopologicalSpace β] …
      inst✝ : (AlgebraicGeometry.topologically fun {α β} [TopologicalSpace α] [Topol …
      hP₂ : ∀ {α β : Type u} [inst : TopologicalSpace α] [inst_1 : TopologicalSpace  …
      hP₃ : ∀ {α β : Type u} [inst : TopologicalSpace α] [inst_1 : TopologicalSpace  …
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      ι : Type u
      U : ι → Y.Opens
      hU : Eq (iSup U) Top.top
      hf : ∀ (i : ι), AlgebraicGeometry.topologically (fun {α β} [TopologicalSpace α …
      ⊢ AlgebraicGeometry.topologically (fun {α β} [TopologicalSpace α] [Topological …
    -/
    apply hP₃ f.base U hU f.base.continuous fun i ↦ ?_
    /-
      P : {α β : Type u} → [inst : TopologicalSpace α] → [inst : TopologicalSpace β] …
      inst✝ : (AlgebraicGeometry.topologically fun {α β} [TopologicalSpace α] [Topol …
      hP₂ : ∀ {α β : Type u} [inst : TopologicalSpace α] [inst_1 : TopologicalSpace  …
      hP₃ : ∀ {α β : Type u} [inst : TopologicalSpace α] [inst_1 : TopologicalSpace  …
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      ι : Type u
      U : ι → Y.Opens
      hU : Eq (iSup U) Top.top
      hf : ∀ (i : ι), AlgebraicGeometry.topologically (fun {α β} [TopologicalSpace α …
      i : ι
      ⊢ P ((U i).carrier.restrictPreimage ⇑f.base)
    -/
    rw [← morphismRestrict_base]
    /-
      P : {α β : Type u} → [inst : TopologicalSpace α] → [inst : TopologicalSpace β] …
      inst✝ : (AlgebraicGeometry.topologically fun {α β} [TopologicalSpace α] [Topol …
      hP₂ : ∀ {α β : Type u} [inst : TopologicalSpace α] [inst_1 : TopologicalSpace  …
      hP₃ : ∀ {α β : Type u} [inst : TopologicalSpace α] [inst_1 : TopologicalSpace  …
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      ι : Type u
      U : ι → Y.Opens
      hU : Eq (iSup U) Top.top
      hf : ∀ (i : ι), AlgebraicGeometry.topologically (fun {α β} [TopologicalSpace α …
      i : ι
      ⊢ P ⇑(AlgebraicGeometry.morphismRestrict f (U i)).base
    -/
    exact hf i
    /-
      🎉 no goals
    -/


/-- A variant of `topologically_isLocalAtTarget`
that takes one iff statement instead of two implications. -/
lemma topologically_isLocalAtTarget'
    [(topologically P).RespectsIso]
    (hP : ∀ {α β : Type u} [TopologicalSpace α] [TopologicalSpace β] (f : α → β) {ι : Type u}
      (U : ι → TopologicalSpace.Opens β) (_ : iSup U = ⊤) (_ : Continuous f),
      P f ↔ (∀ i, P ((U i).carrier.restrictPreimage f))) :
    IsLocalAtTarget (topologically P) := by
  /-
    P : {α β : Type u} → [inst : TopologicalSpace α] → [inst : TopologicalSpace β] …
    inst✝ : (AlgebraicGeometry.topologically fun {α β} [TopologicalSpace α] [Topol …
    hP : ∀ {α β : Type u} [inst : TopologicalSpace α] [inst_1 : TopologicalSpace β …
    ⊢ AlgebraicGeometry.IsLocalAtTarget (AlgebraicGeometry.topologically fun {α β} …
  -/
  refine topologically_isLocalAtTarget P ?_ (fun f _ U hU hU' ↦ (hP f U hU hU').mpr)
  /-
    P : {α β : Type u} → [inst : TopologicalSpace α] → [inst : TopologicalSpace β] …
    inst✝ : (AlgebraicGeometry.topologically fun {α β} [TopologicalSpace α] [Topol …
    hP : ∀ {α β : Type u} [inst : TopologicalSpace α] [inst_1 : TopologicalSpace β …
    ⊢ ∀ {α β : Type u} [inst : TopologicalSpace α] [inst_1 : TopologicalSpace β] ( …
  -/
  introv hf hs H
  /-
    P : {α β : Type u} → [inst : TopologicalSpace α] → [inst : TopologicalSpace β] …
    inst✝² : (AlgebraicGeometry.topologically fun {α β} [TopologicalSpace α] [Topo …
    hP : ∀ {α β : Type u} [inst : TopologicalSpace α] [inst_1 : TopologicalSpace β …
    α β : Type u
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    s : Set β
    hf : Continuous f
    hs : IsOpen s
    H : P f
    ⊢ P (s.restrictPreimage f)
  -/
  have := (hP f (![⊤, Opens.mk s hs] ∘ Equiv.ulift) ?_ hf).mp H ⟨1⟩
    /-
      case refine_2
      P : {α β : Type u} → [inst : TopologicalSpace α] → [inst : TopologicalSpace β] …
      inst✝² : (AlgebraicGeometry.topologically fun {α β} [TopologicalSpace α] [Topo …
      hP : ∀ {α β : Type u} [inst : TopologicalSpace α] [inst_1 : TopologicalSpace β …
      α β : Type u
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      s : Set β
      hf : Continuous f
      hs : IsOpen s
      H : P f
      this : P ((Function.comp (Matrix.vecCons Top.top (Matrix.vecCons { carrier :=  …
      ⊢ P (s.restrictPreimage f)
    -/
  · simpa using this
    /-
      🎉 no goals
    -/
    /-
      case refine_1
      P : {α β : Type u} → [inst : TopologicalSpace α] → [inst : TopologicalSpace β] …
      inst✝² : (AlgebraicGeometry.topologically fun {α β} [TopologicalSpace α] [Topo …
      hP : ∀ {α β : Type u} [inst : TopologicalSpace α] [inst_1 : TopologicalSpace β …
      α β : Type u
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      s : Set β
      hf : Continuous f
      hs : IsOpen s
      H : P f
      ⊢ Eq (iSup (Function.comp (Matrix.vecCons Top.top (Matrix.vecCons { carrier := …
    -/
  · rw [← top_le_iff]
    /-
      case refine_1
      P : {α β : Type u} → [inst : TopologicalSpace α] → [inst : TopologicalSpace β] …
      inst✝² : (AlgebraicGeometry.topologically fun {α β} [TopologicalSpace α] [Topo …
      hP : ∀ {α β : Type u} [inst : TopologicalSpace α] [inst_1 : TopologicalSpace β …
      α β : Type u
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      s : Set β
      hf : Continuous f
      hs : IsOpen s
      H : P f
      ⊢ LE.le Top.top (iSup (Function.comp (Matrix.vecCons Top.top (Matrix.vecCons { …
    -/
    exact le_iSup (![⊤, Opens.mk s hs] ∘ Equiv.ulift) ⟨0⟩
    /-
      🎉 no goals
    -/


/-- `stalkwise P` holds for a morphism if all stalks satisfy `P`. -/
def stalkwise (P : ∀ {R S : Type u} [CommRing R] [CommRing S], (R →+* S) → Prop) :
    MorphismProperty Scheme.{u} :=
  fun _ _ f => ∀ x, P (f.stalkMap x).hom


/-- If `P` respects isos, then `stalkwise P` respects isos. -/
lemma stalkwise_respectsIso (hP : RingHom.RespectsIso P) :
    (stalkwise P).RespectsIso where
  precomp {X Y Z} e (he : IsIso e) f hf := by
    /-
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hP : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
      X Y Z : AlgebraicGeometry.Scheme
      e : Quiver.Hom X Y
      he : CategoryTheory.IsIso e
      f : Quiver.Hom Y Z
      hf : AlgebraicGeometry.stalkwise (fun {R S} [CommRing R] [CommRing S] => P) f
      ⊢ AlgebraicGeometry.stalkwise (fun {R S} [CommRing R] [CommRing S] => P) (Cate …
    -/
    simp only [stalkwise, Scheme.comp_coeBase, TopCat.coe_comp, Function.comp_apply]
    /-
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hP : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
      X Y Z : AlgebraicGeometry.Scheme
      e : Quiver.Hom X Y
      he : CategoryTheory.IsIso e
      f : Quiver.Hom Y Z
      hf : AlgebraicGeometry.stalkwise (fun {R S} [CommRing R] [CommRing S] => P) f
      ⊢ ∀ (x : ↑↑X.toPresheafedSpace), P (AlgebraicGeometry.Scheme.Hom.stalkMap (Cat …
    -/
    intro x
    /-
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hP : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
      X Y Z : AlgebraicGeometry.Scheme
      e : Quiver.Hom X Y
      he : CategoryTheory.IsIso e
      f : Quiver.Hom Y Z
      hf : AlgebraicGeometry.stalkwise (fun {R S} [CommRing R] [CommRing S] => P) f
      x : ↑↑X.toPresheafedSpace
      ⊢ P (AlgebraicGeometry.Scheme.Hom.stalkMap (CategoryTheory.CategoryStruct.comp …
    -/
    rw [Scheme.stalkMap_comp]
    /-
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hP : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
      X Y Z : AlgebraicGeometry.Scheme
      e : Quiver.Hom X Y
      he : CategoryTheory.IsIso e
      f : Quiver.Hom Y Z
      hf : AlgebraicGeometry.stalkwise (fun {R S} [CommRing R] [CommRing S] => P) f
      x : ↑↑X.toPresheafedSpace
      ⊢ P (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Hom.stalkMap …
    -/
    exact (RingHom.RespectsIso.cancel_right_isIso hP _ _).mpr <| hf (e.base x)
    /-
      🎉 no goals
    -/
  postcomp {X Y Z} e (he : IsIso _) f hf := by
    /-
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hP : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
      X Y Z : AlgebraicGeometry.Scheme
      e : Quiver.Hom Y Z
      he : CategoryTheory.IsIso e
      f : Quiver.Hom X Y
      hf : AlgebraicGeometry.stalkwise (fun {R S} [CommRing R] [CommRing S] => P) f
      ⊢ AlgebraicGeometry.stalkwise (fun {R S} [CommRing R] [CommRing S] => P) (Cate …
    -/
    simp only [stalkwise, Scheme.comp_coeBase, TopCat.coe_comp, Function.comp_apply]
    /-
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hP : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
      X Y Z : AlgebraicGeometry.Scheme
      e : Quiver.Hom Y Z
      he : CategoryTheory.IsIso e
      f : Quiver.Hom X Y
      hf : AlgebraicGeometry.stalkwise (fun {R S} [CommRing R] [CommRing S] => P) f
      ⊢ ∀ (x : ↑↑X.toPresheafedSpace), P (AlgebraicGeometry.Scheme.Hom.stalkMap (Cat …
    -/
    intro x
    /-
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hP : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
      X Y Z : AlgebraicGeometry.Scheme
      e : Quiver.Hom Y Z
      he : CategoryTheory.IsIso e
      f : Quiver.Hom X Y
      hf : AlgebraicGeometry.stalkwise (fun {R S} [CommRing R] [CommRing S] => P) f
      x : ↑↑X.toPresheafedSpace
      ⊢ P (AlgebraicGeometry.Scheme.Hom.stalkMap (CategoryTheory.CategoryStruct.comp …
    -/
    rw [Scheme.stalkMap_comp]
    /-
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hP : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
      X Y Z : AlgebraicGeometry.Scheme
      e : Quiver.Hom Y Z
      he : CategoryTheory.IsIso e
      f : Quiver.Hom X Y
      hf : AlgebraicGeometry.stalkwise (fun {R S} [CommRing R] [CommRing S] => P) f
      x : ↑↑X.toPresheafedSpace
      ⊢ P (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Hom.stalkMap …
    -/
    exact (RingHom.RespectsIso.cancel_left_isIso hP _ _).mpr <| hf x
    /-
      🎉 no goals
    -/


/-- If `P` respects isos, then `stalkwise P` is local at the target. -/
lemma stalkwiseIsLocalAtTarget_of_respectsIso (hP : RingHom.RespectsIso P) :
    IsLocalAtTarget (stalkwise P) := by
  have hP' : (RingHom.toMorphismProperty P).RespectsIso :=
    RingHom.toMorphismProperty_respectsIso_iff.mp hP
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hP : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
    hP' : (RingHom.toMorphismProperty fun {R S} [CommRing R] [CommRing S] => P).Re …
    ⊢ AlgebraicGeometry.IsLocalAtTarget (AlgebraicGeometry.stalkwise fun {R S} [Co …
  -/
  letI := stalkwise_respectsIso hP
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hP : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
    hP' : (RingHom.toMorphismProperty fun {R S} [CommRing R] [CommRing S] => P).Re …
    this : (AlgebraicGeometry.stalkwise fun {R S} [CommRing R] [CommRing S] => P). …
    ⊢ AlgebraicGeometry.IsLocalAtTarget (AlgebraicGeometry.stalkwise fun {R S} [Co …
  -/
  apply IsLocalAtTarget.mk'
    /-
      case restrict
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hP : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
      hP' : (RingHom.toMorphismProperty fun {R S} [CommRing R] [CommRing S] => P).Re …
      this : (AlgebraicGeometry.stalkwise fun {R S} [CommRing R] [CommRing S] => P). …
      ⊢ ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y) (U : Y.Opens), Algeb …
    -/
  · intro X Y f U hf x
    apply ((RingHom.toMorphismProperty P).arrow_mk_iso_iff <|
      morphismRestrictStalkMap f U x).mpr <| hf _
    /-
      case of_sSup_eq_top
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hP : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
      hP' : (RingHom.toMorphismProperty fun {R S} [CommRing R] [CommRing S] => P).Re …
      this : (AlgebraicGeometry.stalkwise fun {R S} [CommRing R] [CommRing S] => P). …
      ⊢ ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y) {ι : Type u} (U : ι  …
    -/
  · intro X Y f ι U hU hf x
    /-
      case of_sSup_eq_top
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hP : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
      hP' : (RingHom.toMorphismProperty fun {R S} [CommRing R] [CommRing S] => P).Re …
      this : (AlgebraicGeometry.stalkwise fun {R S} [CommRing R] [CommRing S] => P). …
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      ι : Type u
      U : ι → Y.Opens
      hU : Eq (iSup U) Top.top
      hf : ∀ (i : ι), AlgebraicGeometry.stalkwise (fun {R S} [CommRing R] [CommRing  …
      x : ↑↑X.toPresheafedSpace
      ⊢ P (AlgebraicGeometry.Scheme.Hom.stalkMap f x).hom
    -/
    have hy : f.base x ∈ iSup U := by rw [hU]; trivial
    /-
      case of_sSup_eq_top
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hP : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
      hP' : (RingHom.toMorphismProperty fun {R S} [CommRing R] [CommRing S] => P).Re …
      this : (AlgebraicGeometry.stalkwise fun {R S} [CommRing R] [CommRing S] => P). …
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      ι : Type u
      U : ι → Y.Opens
      hU : Eq (iSup U) Top.top
      hf : ∀ (i : ι), AlgebraicGeometry.stalkwise (fun {R S} [CommRing R] [CommRing  …
      x : ↑↑X.toPresheafedSpace
      hy : Membership.mem (iSup U) (f.base x)
      ⊢ P (AlgebraicGeometry.Scheme.Hom.stalkMap f x).hom
    -/
    obtain ⟨i, hi⟩ := Opens.mem_iSup.mp hy
    exact ((RingHom.toMorphismProperty P).arrow_mk_iso_iff <|
      morphismRestrictStalkMap f (U i) ⟨x, hi⟩).mp <| hf i ⟨x, hi⟩


/-- If `P` respects isos, then `stalkwise P` is local at the source. -/
lemma stalkwise_isLocalAtSource_of_respectsIso (hP : RingHom.RespectsIso P) :
    IsLocalAtSource (stalkwise P) := by
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hP : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
    ⊢ AlgebraicGeometry.IsLocalAtSource (AlgebraicGeometry.stalkwise fun {R S} [Co …
  -/
  letI := stalkwise_respectsIso hP
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hP : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
    this : (AlgebraicGeometry.stalkwise fun {R S} [CommRing R] [CommRing S] => P). …
    ⊢ AlgebraicGeometry.IsLocalAtSource (AlgebraicGeometry.stalkwise fun {R S} [Co …
  -/
  apply IsLocalAtSource.mk'
    /-
      case restrict
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hP : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
      this : (AlgebraicGeometry.stalkwise fun {R S} [CommRing R] [CommRing S] => P). …
      ⊢ ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y) (U : X.Opens), Algeb …
    -/
  · intro X Y f U hf x
    /-
      case restrict
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hP : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
      this : (AlgebraicGeometry.stalkwise fun {R S} [CommRing R] [CommRing S] => P). …
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      U : X.Opens
      hf : AlgebraicGeometry.stalkwise (fun {R S} [CommRing R] [CommRing S] => P) f
      x : ↑↑(↑U).toPresheafedSpace
      ⊢ P (AlgebraicGeometry.Scheme.Hom.stalkMap (CategoryTheory.CategoryStruct.comp …
    -/
    rw [Scheme.stalkMap_comp, CommRingCat.hom_comp, hP.cancel_right_isIso]
    /-
      case restrict
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hP : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
      this : (AlgebraicGeometry.stalkwise fun {R S} [CommRing R] [CommRing S] => P). …
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      U : X.Opens
      hf : AlgebraicGeometry.stalkwise (fun {R S} [CommRing R] [CommRing S] => P) f
      x : ↑↑(↑U).toPresheafedSpace
      ⊢ P (AlgebraicGeometry.Scheme.Hom.stalkMap f (U.ι.base x)).hom
    -/
    exact hf _
    /-
      🎉 no goals
    -/
    /-
      case of_sSup_eq_top
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hP : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
      this : (AlgebraicGeometry.stalkwise fun {R S} [CommRing R] [CommRing S] => P). …
      ⊢ ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y) {ι : Type u} (U : ι  …
    -/
  · intro X Y f ι U hU hf x
    /-
      case of_sSup_eq_top
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hP : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
      this : (AlgebraicGeometry.stalkwise fun {R S} [CommRing R] [CommRing S] => P). …
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      ι : Type u
      U : ι → X.Opens
      hU : Eq (iSup U) Top.top
      hf : ∀ (i : ι), AlgebraicGeometry.stalkwise (fun {R S} [CommRing R] [CommRing  …
      x : ↑↑X.toPresheafedSpace
      ⊢ P (AlgebraicGeometry.Scheme.Hom.stalkMap f x).hom
    -/
    have hy : x ∈ iSup U := by rw [hU]; trivial
    /-
      case of_sSup_eq_top
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hP : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
      this : (AlgebraicGeometry.stalkwise fun {R S} [CommRing R] [CommRing S] => P). …
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      ι : Type u
      U : ι → X.Opens
      hU : Eq (iSup U) Top.top
      hf : ∀ (i : ι), AlgebraicGeometry.stalkwise (fun {R S} [CommRing R] [CommRing  …
      x : ↑↑X.toPresheafedSpace
      hy : Membership.mem (iSup U) x
      ⊢ P (AlgebraicGeometry.Scheme.Hom.stalkMap f x).hom
    -/
    obtain ⟨i, hi⟩ := Opens.mem_iSup.mp hy
    /-
      case of_sSup_eq_top.intro
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hP : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
      this : (AlgebraicGeometry.stalkwise fun {R S} [CommRing R] [CommRing S] => P). …
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      ι : Type u
      U : ι → X.Opens
      hU : Eq (iSup U) Top.top
      hf : ∀ (i : ι), AlgebraicGeometry.stalkwise (fun {R S} [CommRing R] [CommRing  …
      x : ↑↑X.toPresheafedSpace
      hy : Membership.mem (iSup U) x
      i : ι
      hi : Membership.mem (U i) x
      ⊢ P (AlgebraicGeometry.Scheme.Hom.stalkMap f x).hom
    -/
    rw [← hP.cancel_right_isIso _ ((U i).ι.stalkMap ⟨x, hi⟩)]
    /-
      case of_sSup_eq_top.intro
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hP : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
      this : (AlgebraicGeometry.stalkwise fun {R S} [CommRing R] [CommRing S] => P). …
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      ι : Type u
      U : ι → X.Opens
      hU : Eq (iSup U) Top.top
      hf : ∀ (i : ι), AlgebraicGeometry.stalkwise (fun {R S} [CommRing R] [CommRing  …
      x : ↑↑X.toPresheafedSpace
      hy : Membership.mem (iSup U) x
      i : ι
      hi : Membership.mem (U i) x
      ⊢ P ((AlgebraicGeometry.Scheme.Hom.stalkMap (U i).ι ⟨x, hi⟩).hom.comp (Algebra …
    -/
    simpa [Scheme.stalkMap_comp] using hf i ⟨x, hi⟩
    /-
      🎉 no goals
    -/


lemma stalkwise_Spec_map_iff (hP : RingHom.RespectsIso P) {R S : CommRingCat} (φ : R ⟶ S) :
    stalkwise P (Spec.map φ) ↔ ∀ (p : Ideal S) (_ : p.IsPrime),
      P (Localization.localRingHom _ p φ.hom rfl) := by
  have hP' : (RingHom.toMorphismProperty P).RespectsIso :=
    RingHom.toMorphismProperty_respectsIso_iff.mp hP
  /-
    P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
    hP : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
    R S : CommRingCat
    φ : Quiver.Hom R S
    hP' : (RingHom.toMorphismProperty fun {R S} [CommRing R] [CommRing S] => P).Re …
    ⊢ Iff (AlgebraicGeometry.stalkwise (fun {R S} [CommRing R] [CommRing S] => P)  …
  -/
  trans ∀ (p : PrimeSpectrum S), P (Localization.localRingHom _ p.asIdeal φ.hom rfl)
  · exact forall_congr' fun p ↦
      (RingHom.toMorphismProperty P).arrow_mk_iso_iff (Scheme.arrowStalkMapSpecIso _ _)
    /-
      P : {R S : Type u} → [inst : CommRing R] → [inst_1 : CommRing S] → RingHom R S …
      hP : RingHom.RespectsIso fun {R S} [CommRing R] [CommRing S] => P
      R S : CommRingCat
      φ : Quiver.Hom R S
      hP' : (RingHom.toMorphismProperty fun {R S} [CommRing R] [CommRing S] => P).Re …
      ⊢ Iff (∀ (p : PrimeSpectrum ↑S), P (Localization.localRingHom (Ideal.comap φ.h …
    -/
  · exact ⟨fun H p hp ↦ H ⟨p, hp⟩, fun H p ↦ H p.1 p.2⟩
    /-
      🎉 no goals
    -/


/-- If `P` is local at the target, to show that `P` is stable under base change, it suffices to
check this for base change along a morphism of affine schemes. -/
lemma isStableUnderBaseChange_of_isStableUnderBaseChangeOnAffine_of_isLocalAtTarget
    (P : MorphismProperty Scheme) [IsLocalAtTarget P]
    (hP₂ : (of P).IsStableUnderBaseChange) :
    P.IsStableUnderBaseChange :=
  letI := HasAffineProperty.of_isLocalAtTarget P
  HasAffineProperty.isStableUnderBaseChange hP₂


@[deprecated (since := "2024-06-22")]
alias diagonalTargetAffineLocallyOfOpenCover := HasAffineProperty.diagonal_of_openCover


@[deprecated (since := "2024-06-22")]
alias AffineTargetMorphismProperty.diagonalOfTargetAffineLocally :=
  HasAffineProperty.diagonal_of_diagonal_of_isPullback


@[deprecated (since := "2024-06-22")]
alias universallyIsLocalAtTarget := universally_isLocalAtTarget


@[deprecated (since := "2024-06-22")]
alias universallyIsLocalAtTargetOfMorphismRestrict :=
  universally_isLocalAtTarget


