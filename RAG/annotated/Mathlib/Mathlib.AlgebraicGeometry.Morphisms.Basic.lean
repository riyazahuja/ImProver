/--
We say that `P : MorphismProperty Scheme` is local at the target if
1. `P` respects isomorphisms.
2. `P` holds for `f ∣_ U` for an open cover `U` of `Y` if and only if `P` holds for `f`.
Also see `IsLocalAtTarget.mk'` for a convenient constructor.
-/
class IsLocalAtTarget (P : MorphismProperty Scheme) : Prop where
  /-- `P` respects isomorphisms. -/
  respectsIso : P.RespectsIso := by infer_instance
  /-- `P` holds for `f ∣_ U` for an open cover `U` of `Y` if and only if `P` holds for `f`. -/
  iff_of_openCover' :
    ∀ {X Y : Scheme.{u}} (f : X ⟶ Y) (𝒰 : Scheme.OpenCover.{u} Y),
      P f ↔ ∀ i, P (𝒰.pullbackHom f i)


/--
`P` is local at the target if
1. `P` respects isomorphisms.
2. If `P` holds for `f : X ⟶ Y`, then `P` holds for `f ∣_ U` for any `U`.
3. If `P` holds for `f ∣_ U` for an open cover `U` of `Y`, then `P` holds for `f`.
-/
protected lemma mk' {P : MorphismProperty Scheme} [P.RespectsIso]
    (restrict : ∀ {X Y : Scheme} (f : X ⟶ Y) (U : Y.Opens), P f → P (f ∣_ U))
    (of_sSup_eq_top :
      ∀ {X Y : Scheme.{u}} (f : X ⟶ Y) {ι : Type u} (U : ι → Y.Opens), iSup U = ⊤ →
        (∀ i, P (f ∣_ U i)) → P f) :
    IsLocalAtTarget P := by
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    inst✝ : P.RespectsIso
    restrict : ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y) (U : Y.Open …
    of_sSup_eq_top : ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y) {ι :  …
    ⊢ AlgebraicGeometry.IsLocalAtTarget P
  -/
  refine ⟨inferInstance, fun {X Y} f 𝒰 ↦ ⟨?_, fun H ↦ of_sSup_eq_top f _ 𝒰.iSup_opensRange ?_⟩⟩
    /-
      case refine_1
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      inst✝ : P.RespectsIso
      restrict : ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y) (U : Y.Open …
      of_sSup_eq_top : ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y) {ι :  …
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      𝒰 : Y.OpenCover
      ⊢ P f → ∀ (i : 𝒰.1), P (AlgebraicGeometry.Scheme.Cover.pullbackHom 𝒰 f i)
    -/
  · exact fun H i ↦ (P.arrow_mk_iso_iff (morphismRestrictOpensRange f _)).mp (restrict _ _ H)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      inst✝ : P.RespectsIso
      restrict : ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y) (U : Y.Open …
      of_sSup_eq_top : ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y) {ι :  …
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      𝒰 : Y.OpenCover
      H : ∀ (i : 𝒰.1), P (AlgebraicGeometry.Scheme.Cover.pullbackHom 𝒰 f i)
      ⊢ ∀ (i : 𝒰.J), P (AlgebraicGeometry.morphismRestrict f (AlgebraicGeometry.Sche …
    -/
  · exact fun i ↦ (P.arrow_mk_iso_iff (morphismRestrictOpensRange f _)).mpr (H i)
    /-
      🎉 no goals
    -/


/-- The intersection of two morphism properties that are local at the target is again local at
the target. -/
instance inf (P Q : MorphismProperty Scheme) [IsLocalAtTarget P] [IsLocalAtTarget Q] :
    IsLocalAtTarget (P ⊓ Q) where
  iff_of_openCover' {_ _} f 𝒰 :=
    ⟨fun h i ↦ ⟨(iff_of_openCover' f 𝒰).mp h.left i, (iff_of_openCover' f 𝒰).mp h.right i⟩,
     fun h ↦ ⟨(iff_of_openCover' f 𝒰).mpr (fun i ↦ (h i).left),
      (iff_of_openCover' f 𝒰).mpr (fun i ↦ (h i).right)⟩⟩


lemma of_isPullback {UX UY : Scheme.{u}} {iY : UY ⟶ Y} [IsOpenImmersion iY]
    {iX : UX ⟶ X} {f' : UX ⟶ UY} (h : IsPullback iX f' f iY) (H : P f) : P f' := by
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    hP : AlgebraicGeometry.IsLocalAtTarget P
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    UX UY : AlgebraicGeometry.Scheme
    iY : Quiver.Hom UY Y
    inst✝ : AlgebraicGeometry.IsOpenImmersion iY
    iX : Quiver.Hom UX X
    f' : Quiver.Hom UX UY
    h : CategoryTheory.IsPullback iX f' f iY
    H : P f
    ⊢ P f'
  -/
  rw [← P.cancel_left_of_respectsIso h.isoPullback.inv, h.isoPullback_inv_snd]
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    hP : AlgebraicGeometry.IsLocalAtTarget P
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    UX UY : AlgebraicGeometry.Scheme
    iY : Quiver.Hom UY Y
    inst✝ : AlgebraicGeometry.IsOpenImmersion iY
    iX : Quiver.Hom UX X
    f' : Quiver.Hom UX UY
    h : CategoryTheory.IsPullback iX f' f iY
    H : P f
    ⊢ P (CategoryTheory.Limits.pullback.snd f iY)
  -/
  exact (iff_of_openCover' f (Y.affineCover.add iY)).mp H .none
  /-
    🎉 no goals
  -/


theorem restrict (hf : P f) (U : Y.Opens) : P (f ∣_ U) :=
  of_isPullback (isPullback_morphismRestrict f U).flip hf


lemma of_iSup_eq_top {ι} (U : ι → Y.Opens) (hU : iSup U = ⊤)
    (H : ∀ i, P (f ∣_ U i)) : P f := by
  refine (IsLocalAtTarget.iff_of_openCover' f
    (Y.openCoverOfISupEqTop (s := Set.range U) Subtype.val (by ext; simp [← hU]))).mpr fun i ↦ ?_
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    hP : AlgebraicGeometry.IsLocalAtTarget P
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    ι : Sort u_1
    U : ι → Y.Opens
    hU : Eq (iSup U) Top.top
    H : ∀ (i : ι), P (AlgebraicGeometry.morphismRestrict f (U i))
    i : (Y.openCoverOfISupEqTop Subtype.val ⋯).1
    ⊢ P (AlgebraicGeometry.Scheme.Cover.pullbackHom (Y.openCoverOfISupEqTop Subtyp …
  -/
  obtain ⟨_, i, rfl⟩ := i
  /-
    case mk.intro
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    hP : AlgebraicGeometry.IsLocalAtTarget P
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    ι : Sort u_1
    U : ι → Y.Opens
    hU : Eq (iSup U) Top.top
    H : ∀ (i : ι), P (AlgebraicGeometry.morphismRestrict f (U i))
    i : ι
    ⊢ P (AlgebraicGeometry.Scheme.Cover.pullbackHom (Y.openCoverOfISupEqTop Subtyp …
  -/
  refine (P.arrow_mk_iso_iff (morphismRestrictOpensRange f _)).mp ?_
  /-
    case mk.intro
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    hP : AlgebraicGeometry.IsLocalAtTarget P
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    ι : Sort u_1
    U : ι → Y.Opens
    hU : Eq (iSup U) Top.top
    H : ∀ (i : ι), P (AlgebraicGeometry.morphismRestrict f (U i))
    i : ι
    ⊢ P (AlgebraicGeometry.morphismRestrict f (AlgebraicGeometry.Scheme.Hom.opensR …
  -/
  show P (f ∣_ (U i).ι.opensRange)
  /-
    case mk.intro
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    hP : AlgebraicGeometry.IsLocalAtTarget P
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    ι : Sort u_1
    U : ι → Y.Opens
    hU : Eq (iSup U) Top.top
    H : ∀ (i : ι), P (AlgebraicGeometry.morphismRestrict f (U i))
    i : ι
    ⊢ P (AlgebraicGeometry.morphismRestrict f (AlgebraicGeometry.Scheme.Hom.opensR …
  -/
  rw [Scheme.Opens.opensRange_ι]
  /-
    case mk.intro
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    hP : AlgebraicGeometry.IsLocalAtTarget P
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    ι : Sort u_1
    U : ι → Y.Opens
    hU : Eq (iSup U) Top.top
    H : ∀ (i : ι), P (AlgebraicGeometry.morphismRestrict f (U i))
    i : ι
    ⊢ P (AlgebraicGeometry.morphismRestrict f (U i))
  -/
  exact H i
  /-
    🎉 no goals
  -/


theorem iff_of_iSup_eq_top {ι} (U : ι → Y.Opens) (hU : iSup U = ⊤) :
    P f ↔ ∀ i, P (f ∣_ U i) :=
  ⟨fun H _ ↦ restrict H _, of_iSup_eq_top U hU⟩


lemma of_openCover (H : ∀ i, P (𝒰.pullbackHom f i)) : P f := by
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    hP : AlgebraicGeometry.IsLocalAtTarget P
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    𝒰 : Y.OpenCover
    H : ∀ (i : 𝒰.1), P (AlgebraicGeometry.Scheme.Cover.pullbackHom 𝒰 f i)
    ⊢ P f
  -/
  apply of_iSup_eq_top (fun i ↦ (𝒰.map i).opensRange) 𝒰.iSup_opensRange
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    hP : AlgebraicGeometry.IsLocalAtTarget P
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    𝒰 : Y.OpenCover
    H : ∀ (i : 𝒰.1), P (AlgebraicGeometry.Scheme.Cover.pullbackHom 𝒰 f i)
    ⊢ ∀ (i : 𝒰.J), P (AlgebraicGeometry.morphismRestrict f (AlgebraicGeometry.Sche …
  -/
  exact fun i ↦ (P.arrow_mk_iso_iff (morphismRestrictOpensRange f _)).mpr (H i)
  /-
    🎉 no goals
  -/


theorem iff_of_openCover (𝒰 : Y.OpenCover) :
    P f ↔ ∀ i, P (𝒰.pullbackHom f i) :=
  ⟨fun H _ ↦ of_isPullback (.of_hasPullback _ _) H, of_openCover _⟩


/--
We say that `P : MorphismProperty Scheme` is local at the source if
1. `P` respects isomorphisms.
2. `P` holds for `𝒰.map i ≫ f` for an open cover `𝒰` of `X` iff `P` holds for `f : X ⟶ Y`.
Also see `IsLocalAtSource.mk'` for a convenient constructor.
-/
class IsLocalAtSource (P : MorphismProperty Scheme) : Prop where
  /-- `P` respects isomorphisms. -/
  respectsIso : P.RespectsIso := by infer_instance
  /-- `P` holds for `f ∣_ U` for an open cover `U` of `Y` if and only if `P` holds for `f`. -/
  iff_of_openCover' :
    ∀ {X Y : Scheme.{u}} (f : X ⟶ Y) (𝒰 : Scheme.OpenCover.{u} X),
      P f ↔ ∀ i, P (𝒰.map i ≫ f)


/--
`P` is local at the target if
1. `P` respects isomorphisms.
2. If `P` holds for `f : X ⟶ Y`, then `P` holds for `f ∣_ U` for any `U`.
3. If `P` holds for `f ∣_ U` for an open cover `U` of `Y`, then `P` holds for `f`.
-/
protected lemma mk' {P : MorphismProperty Scheme} [P.RespectsIso]
    (restrict : ∀ {X Y : Scheme} (f : X ⟶ Y) (U : X.Opens), P f → P (U.ι ≫ f))
    (of_sSup_eq_top :
      ∀ {X Y : Scheme.{u}} (f : X ⟶ Y) {ι : Type u} (U : ι → X.Opens), iSup U = ⊤ →
        (∀ i, P ((U i).ι ≫ f)) → P f) :
    IsLocalAtSource P := by
  refine ⟨inferInstance, fun {X Y} f 𝒰 ↦
    ⟨fun H i ↦ ?_, fun H ↦ of_sSup_eq_top f _ 𝒰.iSup_opensRange fun i ↦ ?_⟩⟩
  · rw [← IsOpenImmersion.isoOfRangeEq_hom_fac (𝒰.map i) (Scheme.Opens.ι _)
      (congr_arg Opens.carrier (𝒰.map i).opensRange.opensRange_ι.symm), Category.assoc,
      P.cancel_left_of_respectsIso]
    /-
      case refine_1
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      inst✝ : P.RespectsIso
      restrict : ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y) (U : X.Open …
      of_sSup_eq_top : ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y) {ι :  …
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      𝒰 : X.OpenCover
      H : P f
      i : 𝒰.J
      ⊢ P (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Hom.opensRan …
    -/
    exact restrict _ _ H
    /-
      🎉 no goals
    -/
  · rw [← IsOpenImmersion.isoOfRangeEq_inv_fac (𝒰.map i) (Scheme.Opens.ι _)
      (congr_arg Opens.carrier (𝒰.map i).opensRange.opensRange_ι.symm), Category.assoc,
      P.cancel_left_of_respectsIso]
    /-
      case refine_2
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      inst✝ : P.RespectsIso
      restrict : ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y) (U : X.Open …
      of_sSup_eq_top : ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y) {ι :  …
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      𝒰 : X.OpenCover
      H : ∀ (i : 𝒰.J), P (CategoryTheory.CategoryStruct.comp (𝒰.map i) f)
      i : 𝒰.J
      ⊢ P (CategoryTheory.CategoryStruct.comp (𝒰.map i) f)
    -/
    exact H _
    /-
      🎉 no goals
    -/


/-- The intersection of two morphism properties that are local at the target is again local at
the target. -/
instance inf (P Q : MorphismProperty Scheme) [IsLocalAtSource P] [IsLocalAtSource Q] :
    IsLocalAtSource (P ⊓ Q) where
  iff_of_openCover' {_ _} f 𝒰 :=
    ⟨fun h i ↦ ⟨(iff_of_openCover' f 𝒰).mp h.left i, (iff_of_openCover' f 𝒰).mp h.right i⟩,
     fun h ↦ ⟨(iff_of_openCover' f 𝒰).mpr (fun i ↦ (h i).left),
      (iff_of_openCover' f 𝒰).mpr (fun i ↦ (h i).right)⟩⟩


lemma comp {UX : Scheme.{u}} (H : P f) (i : UX ⟶ X) [IsOpenImmersion i] :
    P (i ≫ f) :=
                        /-
                          P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
                          inst✝¹ : AlgebraicGeometry.IsLocalAtSource P
                          X Y : AlgebraicGeometry.Scheme
                          f : Quiver.Hom X Y
                          UX : AlgebraicGeometry.Scheme
                          H : P f
                          i : Quiver.Hom UX X
                          inst✝ : AlgebraicGeometry.IsOpenImmersion i
                          ⊢ AlgebraicGeometry.IsOpenImmersion i
                        -/
  (iff_of_openCover' f (X.affineCover.add i)).mp H .none
                        /-
                          🎉 no goals
                        -/


/-- If `P` is local at the source, then it respects composition on the left with open immersions. -/
instance respectsLeft_isOpenImmersion {P : MorphismProperty Scheme}
    [IsLocalAtSource P] : P.RespectsLeft @IsOpenImmersion where
  precomp i _ _ hf := IsLocalAtSource.comp hf i


lemma of_iSup_eq_top {ι} (U : ι → X.Opens) (hU : iSup U = ⊤)
    (H : ∀ i, P ((U i).ι ≫ f)) : P f := by
  refine (iff_of_openCover' f
    (X.openCoverOfISupEqTop (s := Set.range U) Subtype.val (by ext; simp [← hU]))).mpr fun i ↦ ?_
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsLocalAtSource P
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    ι : Sort u_1
    U : ι → X.Opens
    hU : Eq (iSup U) Top.top
    H : ∀ (i : ι), P (CategoryTheory.CategoryStruct.comp (U i).ι f)
    i : (X.openCoverOfISupEqTop Subtype.val ⋯).J
    ⊢ P (CategoryTheory.CategoryStruct.comp ((X.openCoverOfISupEqTop Subtype.val ⋯ …
  -/
  obtain ⟨_, i, rfl⟩ := i
  /-
    case mk.intro
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsLocalAtSource P
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    ι : Sort u_1
    U : ι → X.Opens
    hU : Eq (iSup U) Top.top
    H : ∀ (i : ι), P (CategoryTheory.CategoryStruct.comp (U i).ι f)
    i : ι
    ⊢ P (CategoryTheory.CategoryStruct.comp ((X.openCoverOfISupEqTop Subtype.val ⋯ …
  -/
  exact H i
  /-
    🎉 no goals
  -/


theorem iff_of_iSup_eq_top {ι} (U : ι → X.Opens) (hU : iSup U = ⊤) :
    P f ↔ ∀ i, P ((U i).ι ≫ f) :=
  ⟨fun H _ ↦ comp H _, of_iSup_eq_top U hU⟩


lemma of_openCover (H : ∀ i, P (𝒰.map i ≫ f)) : P f := by
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsLocalAtSource P
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    𝒰 : X.OpenCover
    H : ∀ (i : 𝒰.J), P (CategoryTheory.CategoryStruct.comp (𝒰.map i) f)
    ⊢ P f
  -/
  refine of_iSup_eq_top (fun i ↦ (𝒰.map i).opensRange) 𝒰.iSup_opensRange fun i ↦ ?_
  rw [← IsOpenImmersion.isoOfRangeEq_inv_fac (𝒰.map i) (Scheme.Opens.ι _)
    (congr_arg Opens.carrier (𝒰.map i).opensRange.opensRange_ι.symm), Category.assoc,
    P.cancel_left_of_respectsIso]
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsLocalAtSource P
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    𝒰 : X.OpenCover
    H : ∀ (i : 𝒰.J), P (CategoryTheory.CategoryStruct.comp (𝒰.map i) f)
    i : 𝒰.J
    ⊢ P (CategoryTheory.CategoryStruct.comp (𝒰.map i) f)
  -/
  exact H i
  /-
    🎉 no goals
  -/


theorem iff_of_openCover :
    P f ↔ ∀ i, P (𝒰.map i ≫ f) :=
  ⟨fun H _ ↦ comp H _, of_openCover _⟩


variable (f) in
lemma of_isOpenImmersion [P.ContainsIdentities] [IsOpenImmersion f] : P f :=
  Category.comp_id f ▸ comp (P.id_mem Y) f


lemma isLocalAtTarget [P.IsMultiplicative]
    (hP : ∀ {X Y Z : Scheme.{u}} (f : X ⟶ Y) (g : Y ⟶ Z) [IsOpenImmersion g], P (f ≫ g) → P f) :
    IsLocalAtTarget P where
  iff_of_openCover' {X Y} f 𝒰 := by
    /-
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      inst✝¹ : AlgebraicGeometry.IsLocalAtSource P
      inst✝ : P.IsMultiplicative
      hP : ∀ {X Y Z : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y) (g : Quiver.Hom …
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      𝒰 : Y.OpenCover
      ⊢ Iff (P f) (∀ (i : 𝒰.1), P (AlgebraicGeometry.Scheme.Cover.pullbackHom 𝒰 f i))
    -/
    refine (iff_of_openCover (𝒰.pullbackCover f)).trans (forall_congr' fun i ↦ ?_)
    /-
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      inst✝¹ : AlgebraicGeometry.IsLocalAtSource P
      inst✝ : P.IsMultiplicative
      hP : ∀ {X Y Z : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y) (g : Quiver.Hom …
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      𝒰 : Y.OpenCover
      i : (AlgebraicGeometry.Scheme.Cover.pullbackCover 𝒰 f).J
      ⊢ Iff (P (CategoryTheory.CategoryStruct.comp ((AlgebraicGeometry.Scheme.Cover. …
    -/
    rw [← Scheme.Cover.pullbackHom_map]
    /-
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      inst✝¹ : AlgebraicGeometry.IsLocalAtSource P
      inst✝ : P.IsMultiplicative
      hP : ∀ {X Y Z : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y) (g : Quiver.Hom …
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      𝒰 : Y.OpenCover
      i : (AlgebraicGeometry.Scheme.Cover.pullbackCover 𝒰 f).J
      ⊢ Iff (P (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Cover.p …
    -/
    constructor
      /-
        case mp
        P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
        inst✝¹ : AlgebraicGeometry.IsLocalAtSource P
        inst✝ : P.IsMultiplicative
        hP : ∀ {X Y Z : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y) (g : Quiver.Hom …
        X Y : AlgebraicGeometry.Scheme
        f : Quiver.Hom X Y
        𝒰 : Y.OpenCover
        i : (AlgebraicGeometry.Scheme.Cover.pullbackCover 𝒰 f).J
        ⊢ P (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Cover.pullba …
      -/
    · exact hP _ _
      /-
        🎉 no goals
      -/
      /-
        case mpr
        P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
        inst✝¹ : AlgebraicGeometry.IsLocalAtSource P
        inst✝ : P.IsMultiplicative
        hP : ∀ {X Y Z : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y) (g : Quiver.Hom …
        X Y : AlgebraicGeometry.Scheme
        f : Quiver.Hom X Y
        𝒰 : Y.OpenCover
        i : (AlgebraicGeometry.Scheme.Cover.pullbackCover 𝒰 f).J
        ⊢ P (AlgebraicGeometry.Scheme.Cover.pullbackHom 𝒰 f i) → P (CategoryTheory.Cat …
      -/
    · exact fun H ↦ P.comp_mem _ _ H (of_isOpenImmersion _)
      /-
        🎉 no goals
      -/


/-- If `P` is local at the source and the target, then restriction on both source and target
preserves `P`. -/
lemma resLE [IsLocalAtTarget P] {U : Y.Opens} {V : X.Opens} (e : V ≤ f ⁻¹ᵁ U)
    (hf : P f) : P (f.resLE U V e) :=
  IsLocalAtSource.comp (IsLocalAtTarget.restrict hf U) _


/-- If `P` is local at the source, local at the target and is stable under post-composition with
open immersions, then `P` can be checked locally around points. -/
lemma iff_exists_resLE [IsLocalAtTarget P] [P.RespectsRight @IsOpenImmersion] :
    P f ↔ ∀ x : X, ∃ (U : Y.Opens) (V : X.Opens) (_ : x ∈ V.1) (e : V ≤ f ⁻¹ᵁ U),
      P (f.resLE U V e) := by
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    inst✝² : AlgebraicGeometry.IsLocalAtSource P
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝¹ : AlgebraicGeometry.IsLocalAtTarget P
    inst✝ : P.RespectsRight @AlgebraicGeometry.IsOpenImmersion
    ⊢ Iff (P f) (∀ (x : ↑↑X.toPresheafedSpace), Exists fun U => Exists fun V => Ex …
  -/
  refine ⟨fun hf x ↦ ⟨⊤, ⊤, trivial, by simp, resLE _ hf⟩, fun hf ↦ ?_⟩
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    inst✝² : AlgebraicGeometry.IsLocalAtSource P
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝¹ : AlgebraicGeometry.IsLocalAtTarget P
    inst✝ : P.RespectsRight @AlgebraicGeometry.IsOpenImmersion
    hf : ∀ (x : ↑↑X.toPresheafedSpace), Exists fun U => Exists fun V => Exists fun …
    ⊢ P f
  -/
  choose U V hxU e hf using hf
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    inst✝² : AlgebraicGeometry.IsLocalAtSource P
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝¹ : AlgebraicGeometry.IsLocalAtTarget P
    inst✝ : P.RespectsRight @AlgebraicGeometry.IsOpenImmersion
    U : ↑↑X.toPresheafedSpace → Y.Opens
    V : ↑↑X.toPresheafedSpace → X.Opens
    hxU : ∀ (x : ↑↑X.toPresheafedSpace), Membership.mem (V x).carrier x
    e : ∀ (x : ↑↑X.toPresheafedSpace), LE.le (V x) ((TopologicalSpace.Opens.map f. …
    hf : ∀ (x : ↑↑X.toPresheafedSpace), P (AlgebraicGeometry.Scheme.Hom.resLE f (U …
    ⊢ P f
  -/
  rw [IsLocalAtSource.iff_of_iSup_eq_top (fun x : X ↦ V x) (P := P)]
    /-
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      inst✝² : AlgebraicGeometry.IsLocalAtSource P
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      inst✝¹ : AlgebraicGeometry.IsLocalAtTarget P
      inst✝ : P.RespectsRight @AlgebraicGeometry.IsOpenImmersion
      U : ↑↑X.toPresheafedSpace → Y.Opens
      V : ↑↑X.toPresheafedSpace → X.Opens
      hxU : ∀ (x : ↑↑X.toPresheafedSpace), Membership.mem (V x).carrier x
      e : ∀ (x : ↑↑X.toPresheafedSpace), LE.le (V x) ((TopologicalSpace.Opens.map f. …
      hf : ∀ (x : ↑↑X.toPresheafedSpace), P (AlgebraicGeometry.Scheme.Hom.resLE f (U …
      ⊢ ∀ (i : ↑↑X.toPresheafedSpace), P (CategoryTheory.CategoryStruct.comp (V i).ι …
    -/
  · intro x
    /-
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      inst✝² : AlgebraicGeometry.IsLocalAtSource P
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      inst✝¹ : AlgebraicGeometry.IsLocalAtTarget P
      inst✝ : P.RespectsRight @AlgebraicGeometry.IsOpenImmersion
      U : ↑↑X.toPresheafedSpace → Y.Opens
      V : ↑↑X.toPresheafedSpace → X.Opens
      hxU : ∀ (x : ↑↑X.toPresheafedSpace), Membership.mem (V x).carrier x
      e : ∀ (x : ↑↑X.toPresheafedSpace), LE.le (V x) ((TopologicalSpace.Opens.map f. …
      hf : ∀ (x : ↑↑X.toPresheafedSpace), P (AlgebraicGeometry.Scheme.Hom.resLE f (U …
      x : ↑↑X.toPresheafedSpace
      ⊢ P (CategoryTheory.CategoryStruct.comp (V x).ι f)
    -/
    rw [← Scheme.Hom.resLE_comp_ι _ (e x)]
    /-
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      inst✝² : AlgebraicGeometry.IsLocalAtSource P
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      inst✝¹ : AlgebraicGeometry.IsLocalAtTarget P
      inst✝ : P.RespectsRight @AlgebraicGeometry.IsOpenImmersion
      U : ↑↑X.toPresheafedSpace → Y.Opens
      V : ↑↑X.toPresheafedSpace → X.Opens
      hxU : ∀ (x : ↑↑X.toPresheafedSpace), Membership.mem (V x).carrier x
      e : ∀ (x : ↑↑X.toPresheafedSpace), LE.le (V x) ((TopologicalSpace.Opens.map f. …
      hf : ∀ (x : ↑↑X.toPresheafedSpace), P (AlgebraicGeometry.Scheme.Hom.resLE f (U …
      x : ↑↑X.toPresheafedSpace
      ⊢ P (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Hom.resLE f  …
    -/
    exact MorphismProperty.RespectsRight.postcomp (Q := @IsOpenImmersion) _ inferInstance _ (hf x)
    /-
      🎉 no goals
    -/
    /-
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      inst✝² : AlgebraicGeometry.IsLocalAtSource P
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      inst✝¹ : AlgebraicGeometry.IsLocalAtTarget P
      inst✝ : P.RespectsRight @AlgebraicGeometry.IsOpenImmersion
      U : ↑↑X.toPresheafedSpace → Y.Opens
      V : ↑↑X.toPresheafedSpace → X.Opens
      hxU : ∀ (x : ↑↑X.toPresheafedSpace), Membership.mem (V x).carrier x
      e : ∀ (x : ↑↑X.toPresheafedSpace), LE.le (V x) ((TopologicalSpace.Opens.map f. …
      hf : ∀ (x : ↑↑X.toPresheafedSpace), P (AlgebraicGeometry.Scheme.Hom.resLE f (U …
      ⊢ Eq (iSup fun x => V x) Top.top
    -/
  · rw [eq_top_iff]
    /-
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      inst✝² : AlgebraicGeometry.IsLocalAtSource P
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      inst✝¹ : AlgebraicGeometry.IsLocalAtTarget P
      inst✝ : P.RespectsRight @AlgebraicGeometry.IsOpenImmersion
      U : ↑↑X.toPresheafedSpace → Y.Opens
      V : ↑↑X.toPresheafedSpace → X.Opens
      hxU : ∀ (x : ↑↑X.toPresheafedSpace), Membership.mem (V x).carrier x
      e : ∀ (x : ↑↑X.toPresheafedSpace), LE.le (V x) ((TopologicalSpace.Opens.map f. …
      hf : ∀ (x : ↑↑X.toPresheafedSpace), P (AlgebraicGeometry.Scheme.Hom.resLE f (U …
      ⊢ LE.le Top.top (iSup fun x => V x)
    -/
    rintro x -
    /-
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      inst✝² : AlgebraicGeometry.IsLocalAtSource P
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      inst✝¹ : AlgebraicGeometry.IsLocalAtTarget P
      inst✝ : P.RespectsRight @AlgebraicGeometry.IsOpenImmersion
      U : ↑↑X.toPresheafedSpace → Y.Opens
      V : ↑↑X.toPresheafedSpace → X.Opens
      hxU : ∀ (x : ↑↑X.toPresheafedSpace), Membership.mem (V x).carrier x
      e : ∀ (x : ↑↑X.toPresheafedSpace), LE.le (V x) ((TopologicalSpace.Opens.map f. …
      hf : ∀ (x : ↑↑X.toPresheafedSpace), P (AlgebraicGeometry.Scheme.Hom.resLE f (U …
      x : ↑↑X.toPresheafedSpace
      ⊢ Membership.mem (↑(iSup fun x => V x)) x
    -/
    simp only [Opens.coe_iSup, Set.mem_iUnion, SetLike.mem_coe]
    /-
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      inst✝² : AlgebraicGeometry.IsLocalAtSource P
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      inst✝¹ : AlgebraicGeometry.IsLocalAtTarget P
      inst✝ : P.RespectsRight @AlgebraicGeometry.IsOpenImmersion
      U : ↑↑X.toPresheafedSpace → Y.Opens
      V : ↑↑X.toPresheafedSpace → X.Opens
      hxU : ∀ (x : ↑↑X.toPresheafedSpace), Membership.mem (V x).carrier x
      e : ∀ (x : ↑↑X.toPresheafedSpace), LE.le (V x) ((TopologicalSpace.Opens.map f. …
      hf : ∀ (x : ↑↑X.toPresheafedSpace), P (AlgebraicGeometry.Scheme.Hom.resLE f (U …
      x : ↑↑X.toPresheafedSpace
      ⊢ Exists fun i => Membership.mem (V i) x
    -/
    use x, hxU x
    /-
      🎉 no goals
    -/


/-- An `AffineTargetMorphismProperty` is a class of morphisms from an arbitrary scheme into an
affine scheme. -/
def AffineTargetMorphismProperty :=
  ∀ ⦃X Y : Scheme⦄ (_ : X ⟶ Y) [IsAffine Y], Prop


@[ext]
lemma ext {P Q : AffineTargetMorphismProperty}
    (H : ∀ ⦃X Y : Scheme⦄ (f : X ⟶ Y) [IsAffine Y], P f ↔ Q f) : P = Q := by
  /-
    P Q : AlgebraicGeometry.AffineTargetMorphismProperty
    H : ∀ ⦃X Y : AlgebraicGeometry.Scheme⦄ (f : Quiver.Hom X Y) [inst : AlgebraicG …
    ⊢ Eq P Q
  -/
  delta AffineTargetMorphismProperty; ext; exact H _
                                           /-
                                             🎉 no goals
                                           -/


/-- The restriction of a `MorphismProperty Scheme` to morphisms with affine target. -/
def of (P : MorphismProperty Scheme) : AffineTargetMorphismProperty :=
  fun _ _ f _ ↦ P f


/-- An `AffineTargetMorphismProperty` can be extended to a `MorphismProperty` such that it
*never* holds when the target is not affine -/
def toProperty (P : AffineTargetMorphismProperty) :
    MorphismProperty Scheme := fun _ _ f => ∃ h, @P _ _ f h


theorem toProperty_apply (P : AffineTargetMorphismProperty)
    {X Y : Scheme} (f : X ⟶ Y) [i : IsAffine Y] : P.toProperty f ↔ P f := by
  /-
    P : AlgebraicGeometry.AffineTargetMorphismProperty
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    i : AlgebraicGeometry.IsAffine Y
    ⊢ Iff (P.toProperty f) (P f)
  -/
  delta AffineTargetMorphismProperty.toProperty; simp [*]
                                                 /-
                                                   🎉 no goals
                                                 -/


theorem cancel_left_of_respectsIso
    (P : AffineTargetMorphismProperty) [P.toProperty.RespectsIso]
    {X Y Z : Scheme} (f : X ⟶ Y) (g : Y ⟶ Z) [IsIso f] [IsAffine Z] : P (f ≫ g) ↔ P g := by
  /-
    P : AlgebraicGeometry.AffineTargetMorphismProperty
    inst✝² : P.toProperty.RespectsIso
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    inst✝¹ : CategoryTheory.IsIso f
    inst✝ : AlgebraicGeometry.IsAffine Z
    ⊢ Iff (P (CategoryTheory.CategoryStruct.comp f g)) (P g)
  -/
  rw [← P.toProperty_apply, ← P.toProperty_apply, P.toProperty.cancel_left_of_respectsIso]
  /-
    🎉 no goals
  -/


theorem cancel_right_of_respectsIso
    (P : AffineTargetMorphismProperty) [P.toProperty.RespectsIso]
    {X Y Z : Scheme} (f : X ⟶ Y) (g : Y ⟶ Z) [IsIso g] [IsAffine Z] [IsAffine Y] :
    P (f ≫ g) ↔ P f := by rw [← P.toProperty_apply, ← P.toProperty_apply,
      P.toProperty.cancel_right_of_respectsIso]


@[deprecated (since := "2024-07-02")] alias affine_cancel_left_isIso :=
  AffineTargetMorphismProperty.cancel_left_of_respectsIso

@[deprecated (since := "2024-07-02")] alias affine_cancel_right_isIso :=
  AffineTargetMorphismProperty.cancel_right_of_respectsIso


theorem arrow_mk_iso_iff
    (P : AffineTargetMorphismProperty) [P.toProperty.RespectsIso]
    {X Y X' Y' : Scheme} {f : X ⟶ Y} {f' : X' ⟶ Y'}
    (e : Arrow.mk f ≅ Arrow.mk f') {h : IsAffine Y} :
    letI : IsAffine Y' := isAffine_of_isIso (Y := Y) e.inv.right
    P f ↔ P f' := by
  /-
    P : AlgebraicGeometry.AffineTargetMorphismProperty
    inst✝ : P.toProperty.RespectsIso
    X Y X' Y' : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    f' : Quiver.Hom X' Y'
    e : CategoryTheory.Iso (CategoryTheory.Arrow.mk f) (CategoryTheory.Arrow.mk f')
    h : AlgebraicGeometry.IsAffine Y
    ⊢ Iff (P f) (P f')
  -/
  rw [← P.toProperty_apply, ← P.toProperty_apply, P.toProperty.arrow_mk_iso_iff e]
  /-
    🎉 no goals
  -/


theorem respectsIso_mk {P : AffineTargetMorphismProperty}
    (h₁ : ∀ {X Y Z} (e : X ≅ Y) (f : Y ⟶ Z) [IsAffine Z], P f → P (e.hom ≫ f))
    (h₂ : ∀ {X Y Z} (e : Y ≅ Z) (f : X ⟶ Y) [IsAffine Y],
      P f → @P _ _ (f ≫ e.hom) (isAffine_of_isIso e.inv)) :
    P.toProperty.RespectsIso := by
  /-
    P : AlgebraicGeometry.AffineTargetMorphismProperty
    h₁ : ∀ {X Y Z : AlgebraicGeometry.Scheme} (e : CategoryTheory.Iso X Y) (f : Qu …
    h₂ : ∀ {X Y Z : AlgebraicGeometry.Scheme} (e : CategoryTheory.Iso Y Z) (f : Qu …
    ⊢ P.toProperty.RespectsIso
  -/
  apply MorphismProperty.RespectsIso.mk
    /-
      case hprecomp
      P : AlgebraicGeometry.AffineTargetMorphismProperty
      h₁ : ∀ {X Y Z : AlgebraicGeometry.Scheme} (e : CategoryTheory.Iso X Y) (f : Qu …
      h₂ : ∀ {X Y Z : AlgebraicGeometry.Scheme} (e : CategoryTheory.Iso Y Z) (f : Qu …
      ⊢ ∀ {X Y Z : AlgebraicGeometry.Scheme} (e : CategoryTheory.Iso X Y) (f : Quive …
    -/
  · rintro X Y Z e f ⟨a, h⟩; exact ⟨a, h₁ e f h⟩
                             /-
                               🎉 no goals
                             -/
    /-
      case hpostcomp
      P : AlgebraicGeometry.AffineTargetMorphismProperty
      h₁ : ∀ {X Y Z : AlgebraicGeometry.Scheme} (e : CategoryTheory.Iso X Y) (f : Qu …
      h₂ : ∀ {X Y Z : AlgebraicGeometry.Scheme} (e : CategoryTheory.Iso Y Z) (f : Qu …
      ⊢ ∀ {X Y Z : AlgebraicGeometry.Scheme} (e : CategoryTheory.Iso Y Z) (f : Quive …
    -/
  · rintro X Y Z e f ⟨a, h⟩; exact ⟨isAffine_of_isIso e.inv, h₂ e f h⟩
                             /-
                               🎉 no goals
                             -/


instance respectsIso_of
    (P : MorphismProperty Scheme) [P.RespectsIso] :
    (of P).toProperty.RespectsIso := by
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    inst✝ : P.RespectsIso
    ⊢ (AlgebraicGeometry.AffineTargetMorphismProperty.of P).toProperty.RespectsIso
  -/
  apply respectsIso_mk
    /-
      case h₁
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      inst✝ : P.RespectsIso
      ⊢ ∀ {X Y Z : AlgebraicGeometry.Scheme} (e : CategoryTheory.Iso X Y) (f : Quive …
    -/
  · intro _ _ _ _ _ _; apply MorphismProperty.RespectsIso.precomp
                       /-
                         🎉 no goals
                       -/
    /-
      case h₂
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      inst✝ : P.RespectsIso
      ⊢ ∀ {X Y Z : AlgebraicGeometry.Scheme} (e : CategoryTheory.Iso Y Z) (f : Quive …
    -/
  · intro _ _ _ _ _ _; apply MorphismProperty.RespectsIso.postcomp
                       /-
                         🎉 no goals
                       -/


/-- We say that `P : AffineTargetMorphismProperty` is a local property if
1. `P` respects isomorphisms.
2. If `P` holds for `f : X ⟶ Y`, then `P` holds for `f ∣_ Y.basicOpen r` for any
  global section `r`.
3. If `P` holds for `f ∣_ Y.basicOpen r` for all `r` in a spanning set of the global sections,
  then `P` holds for `f`.
-/
class IsLocal (P : AffineTargetMorphismProperty) : Prop where
  /-- `P` as a morphism property respects isomorphisms -/
  respectsIso : P.toProperty.RespectsIso
  /-- `P` is stable under restriction to basic open set of global sections. -/
  to_basicOpen :
    ∀ {X Y : Scheme} [IsAffine Y] (f : X ⟶ Y) (r : Γ(Y, ⊤)), P f → P (f ∣_ Y.basicOpen r)
  /-- `P` for `f` if `P` holds for `f` restricted to basic sets of a spanning set of the global
    sections -/
  of_basicOpenCover :
    ∀ {X Y : Scheme} [IsAffine Y] (f : X ⟶ Y) (s : Finset Γ(Y, ⊤))
      (_ : Ideal.span (s : Set Γ(Y, ⊤)) = ⊤), (∀ r : s, P (f ∣_ Y.basicOpen r.1)) → P f


open AffineTargetMorphismProperty in
instance (P : MorphismProperty Scheme) [IsLocalAtTarget P] : (of P).IsLocal where
  respectsIso := inferInstance
  to_basicOpen _ _ H := IsLocalAtTarget.restrict H _
  of_basicOpenCover {_ Y} _ _ _ hs := IsLocalAtTarget.of_iSup_eq_top _
    (((isAffineOpen_top Y).basicOpen_union_eq_self_iff _).mpr hs)


/-- A `P : AffineTargetMorphismProperty` is stable under base change if `P` holds for `Y ⟶ S`
implies that `P` holds for `X ×ₛ Y ⟶ X` with `X` and `S` affine schemes. -/
def IsStableUnderBaseChange (P : AffineTargetMorphismProperty) : Prop :=
  ∀ ⦃Z X Y S : Scheme⦄ [IsAffine S] [IsAffine X] {f : X ⟶ S} {g : Y ⟶ S}
    {f' : Z ⟶ Y} {g' : Z ⟶ X}, IsPullback g' f' f g → P g → P g'


lemma IsStableUnderBaseChange.mk (P : AffineTargetMorphismProperty) [P.toProperty.RespectsIso]
    (H : ∀ ⦃X Y S : Scheme⦄ [IsAffine S] [IsAffine X] (f : X ⟶ S) (g : Y ⟶ S),
      P g → P (pullback.fst f g)) : P.IsStableUnderBaseChange := by
  /-
    P : AlgebraicGeometry.AffineTargetMorphismProperty
    inst✝ : P.toProperty.RespectsIso
    H : ∀ ⦃X Y S : AlgebraicGeometry.Scheme⦄ [inst : AlgebraicGeometry.IsAffine S] …
    ⊢ P.IsStableUnderBaseChange
  -/
  intros Z X Y S _ _ f g f' g' h hg
  /-
    P : AlgebraicGeometry.AffineTargetMorphismProperty
    inst✝² : P.toProperty.RespectsIso
    H : ∀ ⦃X Y S : AlgebraicGeometry.Scheme⦄ [inst : AlgebraicGeometry.IsAffine S] …
    Z X Y S : AlgebraicGeometry.Scheme
    inst✝¹ : AlgebraicGeometry.IsAffine S
    inst✝ : AlgebraicGeometry.IsAffine X
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    f' : Quiver.Hom Z Y
    g' : Quiver.Hom Z X
    h : CategoryTheory.IsPullback g' f' f g
    hg : P g
    ⊢ P g'
  -/
  rw [← P.cancel_left_of_respectsIso h.isoPullback.inv, h.isoPullback_inv_fst]
  /-
    P : AlgebraicGeometry.AffineTargetMorphismProperty
    inst✝² : P.toProperty.RespectsIso
    H : ∀ ⦃X Y S : AlgebraicGeometry.Scheme⦄ [inst : AlgebraicGeometry.IsAffine S] …
    Z X Y S : AlgebraicGeometry.Scheme
    inst✝¹ : AlgebraicGeometry.IsAffine S
    inst✝ : AlgebraicGeometry.IsAffine X
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    f' : Quiver.Hom Z Y
    g' : Quiver.Hom Z X
    h : CategoryTheory.IsPullback g' f' f g
    hg : P g
    ⊢ P (CategoryTheory.Limits.pullback.fst f g)
  -/
  exact H f g hg
  /-
    🎉 no goals
  -/


/-- For a `P : AffineTargetMorphismProperty`, `targetAffineLocally P` holds for
`f : X ⟶ Y` whenever `P` holds for the restriction of `f` on every affine open subset of `Y`. -/
def targetAffineLocally (P : AffineTargetMorphismProperty) : MorphismProperty Scheme :=
  fun {X Y : Scheme} (f : X ⟶ Y) => ∀ U : Y.affineOpens, P (f ∣_ U)


theorem of_targetAffineLocally_of_isPullback
    {P : AffineTargetMorphismProperty} [P.IsLocal]
    {X Y UX UY : Scheme.{u}} [IsAffine UY] {f : X ⟶ Y} {iY : UY ⟶ Y} [IsOpenImmersion iY]
    {iX : UX ⟶ X} {f' : UX ⟶ UY} (h : IsPullback iX f' f iY) (hf : targetAffineLocally P f) :
    P f' := by
  /-
    P : AlgebraicGeometry.AffineTargetMorphismProperty
    inst✝² : P.IsLocal
    X Y UX UY : AlgebraicGeometry.Scheme
    inst✝¹ : AlgebraicGeometry.IsAffine UY
    f : Quiver.Hom X Y
    iY : Quiver.Hom UY Y
    inst✝ : AlgebraicGeometry.IsOpenImmersion iY
    iX : Quiver.Hom UX X
    f' : Quiver.Hom UX UY
    h : CategoryTheory.IsPullback iX f' f iY
    hf : AlgebraicGeometry.targetAffineLocally P f
    ⊢ P f'
  -/
  rw [← P.cancel_left_of_respectsIso h.isoPullback.inv, h.isoPullback_inv_snd]
  exact (P.arrow_mk_iso_iff
    (morphismRestrictOpensRange f _)).mp (hf ⟨_, isAffineOpen_opensRange iY⟩)


instance (P : AffineTargetMorphismProperty) [P.toProperty.RespectsIso] :
    (targetAffineLocally P).RespectsIso := by
  /-
    P : AlgebraicGeometry.AffineTargetMorphismProperty
    inst✝ : P.toProperty.RespectsIso
    ⊢ (AlgebraicGeometry.targetAffineLocally P).RespectsIso
  -/
  apply MorphismProperty.RespectsIso.mk
    /-
      case hprecomp
      P : AlgebraicGeometry.AffineTargetMorphismProperty
      inst✝ : P.toProperty.RespectsIso
      ⊢ ∀ {X Y Z : AlgebraicGeometry.Scheme} (e : CategoryTheory.Iso X Y) (f : Quive …
    -/
  · introv H U
    /-
      case hprecomp
      P : AlgebraicGeometry.AffineTargetMorphismProperty
      inst✝ : P.toProperty.RespectsIso
      X Y Z : AlgebraicGeometry.Scheme
      e : CategoryTheory.Iso X Y
      f : Quiver.Hom Y Z
      H : AlgebraicGeometry.targetAffineLocally P f
      U : ↑Z.affineOpens
      ⊢ P (AlgebraicGeometry.morphismRestrict (CategoryTheory.CategoryStruct.comp e. …
    -/
    rw [morphismRestrict_comp, P.cancel_left_of_respectsIso]
    /-
      case hprecomp
      P : AlgebraicGeometry.AffineTargetMorphismProperty
      inst✝ : P.toProperty.RespectsIso
      X Y Z : AlgebraicGeometry.Scheme
      e : CategoryTheory.Iso X Y
      f : Quiver.Hom Y Z
      H : AlgebraicGeometry.targetAffineLocally P f
      U : ↑Z.affineOpens
      ⊢ P (AlgebraicGeometry.morphismRestrict f ↑U)
    -/
    exact H U
    /-
      🎉 no goals
    -/
    /-
      case hpostcomp
      P : AlgebraicGeometry.AffineTargetMorphismProperty
      inst✝ : P.toProperty.RespectsIso
      ⊢ ∀ {X Y Z : AlgebraicGeometry.Scheme} (e : CategoryTheory.Iso Y Z) (f : Quive …
    -/
  · introv H
    /-
      case hpostcomp
      P : AlgebraicGeometry.AffineTargetMorphismProperty
      inst✝ : P.toProperty.RespectsIso
      X Y Z : AlgebraicGeometry.Scheme
      e : CategoryTheory.Iso Y Z
      f : Quiver.Hom X Y
      H : AlgebraicGeometry.targetAffineLocally P f
      ⊢ AlgebraicGeometry.targetAffineLocally P (CategoryTheory.CategoryStruct.comp  …
    -/
    rintro ⟨U, hU : IsAffineOpen U⟩; dsimp
    /-
      case hpostcomp.mk
      P : AlgebraicGeometry.AffineTargetMorphismProperty
      inst✝ : P.toProperty.RespectsIso
      X Y Z : AlgebraicGeometry.Scheme
      e : CategoryTheory.Iso Y Z
      f : Quiver.Hom X Y
      H : AlgebraicGeometry.targetAffineLocally P f
      U : Z.Opens
      hU : AlgebraicGeometry.IsAffineOpen U
      ⊢ P (AlgebraicGeometry.morphismRestrict (CategoryTheory.CategoryStruct.comp f  …
    -/
    haveI : IsAffine _ := hU.preimage_of_isIso e.hom
    /-
      case hpostcomp.mk
      P : AlgebraicGeometry.AffineTargetMorphismProperty
      inst✝ : P.toProperty.RespectsIso
      X Y Z : AlgebraicGeometry.Scheme
      e : CategoryTheory.Iso Y Z
      f : Quiver.Hom X Y
      H : AlgebraicGeometry.targetAffineLocally P f
      U : Z.Opens
      hU : AlgebraicGeometry.IsAffineOpen U
      this : AlgebraicGeometry.IsAffine ↑((TopologicalSpace.Opens.map e.hom.base).ob …
      ⊢ P (AlgebraicGeometry.morphismRestrict (CategoryTheory.CategoryStruct.comp f  …
    -/
    rw [morphismRestrict_comp, P.cancel_right_of_respectsIso]
    /-
      case hpostcomp.mk
      P : AlgebraicGeometry.AffineTargetMorphismProperty
      inst✝ : P.toProperty.RespectsIso
      X Y Z : AlgebraicGeometry.Scheme
      e : CategoryTheory.Iso Y Z
      f : Quiver.Hom X Y
      H : AlgebraicGeometry.targetAffineLocally P f
      U : Z.Opens
      hU : AlgebraicGeometry.IsAffineOpen U
      this : AlgebraicGeometry.IsAffine ↑((TopologicalSpace.Opens.map e.hom.base).ob …
      ⊢ P (AlgebraicGeometry.morphismRestrict f ((TopologicalSpace.Opens.map e.hom.b …
    -/
    exact H ⟨(Opens.map e.hom.base).obj U, hU.preimage_of_isIso e.hom⟩
    /-
      🎉 no goals
    -/


/--
`HasAffineProperty P Q` is a type class asserting that `P` is local at the target, and over affine
schemes, it is equivalent to `Q : AffineTargetMorphismProperty`.
To make the proofs easier, we state it instead as
1. `Q` is local at the target
2. `P f` if and only if `∀ U, Q (f ∣_ U)` ranging over all affine opens of `U`.
See `HasAffineProperty.iff`.
-/
class HasAffineProperty (P : MorphismProperty Scheme)
    (Q : outParam AffineTargetMorphismProperty) : Prop where
  isLocal_affineProperty : Q.IsLocal
  eq_targetAffineLocally' : P = targetAffineLocally Q


instance (Q : AffineTargetMorphismProperty) [Q.IsLocal] :
    HasAffineProperty (targetAffineLocally Q) Q :=
  ⟨inferInstance, rfl⟩


lemma eq_targetAffineLocally : P = targetAffineLocally Q := eq_targetAffineLocally'


/-- Every property local at the target can be associated with an affine target property.
This is not an instance as the associated property can often take on simpler forms. -/
lemma of_isLocalAtTarget (P) [IsLocalAtTarget P] :
    HasAffineProperty P (AffineTargetMorphismProperty.of P) where
  isLocal_affineProperty := inferInstance
  eq_targetAffineLocally' := by
    /-
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      inst✝ : AlgebraicGeometry.IsLocalAtTarget P
      ⊢ Eq P (AlgebraicGeometry.targetAffineLocally (AlgebraicGeometry.AffineTargetM …
    -/
    ext X Y f
    /-
      case h
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      inst✝ : AlgebraicGeometry.IsLocalAtTarget P
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      ⊢ Iff (P f) (AlgebraicGeometry.targetAffineLocally (AlgebraicGeometry.AffineTa …
    -/
    constructor
      /-
        case h.mp
        P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
        inst✝ : AlgebraicGeometry.IsLocalAtTarget P
        X Y : AlgebraicGeometry.Scheme
        f : Quiver.Hom X Y
        ⊢ P f → AlgebraicGeometry.targetAffineLocally (AlgebraicGeometry.AffineTargetM …
      -/
    · intro hf ⟨U, hU⟩
      /-
        case h.mp
        P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
        inst✝ : AlgebraicGeometry.IsLocalAtTarget P
        X Y : AlgebraicGeometry.Scheme
        f : Quiver.Hom X Y
        hf : P f
        U : Y.Opens
        hU : Membership.mem Y.affineOpens U
        ⊢ AlgebraicGeometry.AffineTargetMorphismProperty.of P (AlgebraicGeometry.morph …
      -/
      exact IsLocalAtTarget.restrict hf _
      /-
        🎉 no goals
      -/
      /-
        case h.mpr
        P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
        inst✝ : AlgebraicGeometry.IsLocalAtTarget P
        X Y : AlgebraicGeometry.Scheme
        f : Quiver.Hom X Y
        ⊢ AlgebraicGeometry.targetAffineLocally (AlgebraicGeometry.AffineTargetMorphis …
      -/
    · intro hf
      exact IsLocalAtTarget.of_openCover (P := P) Y.affineCover
        fun i ↦ of_targetAffineLocally_of_isPullback (.of_hasPullback _ _) hf


lemma copy {P P'} {Q Q'} [HasAffineProperty P Q]
    (e : P = P') (e' : Q = Q') : HasAffineProperty P' Q' where
  isLocal_affineProperty := e' ▸ isLocal_affineProperty P
  eq_targetAffineLocally' := e' ▸ e.symm ▸ eq_targetAffineLocally P


theorem of_isPullback {UX UY : Scheme.{u}} [IsAffine UY] {iY : UY ⟶ Y} [IsOpenImmersion iY]
    {iX : UX ⟶ X} {f' : UX ⟶ UY} (h : IsPullback iX f' f iY) (hf : P f) :
    Q f' :=
  letI := isLocal_affineProperty P
  of_targetAffineLocally_of_isPullback h (eq_targetAffineLocally (P := P) ▸ hf)


theorem restrict (hf : P f) (U : Y.affineOpens) :
    Q (f ∣_ U) :=
  of_isPullback (isPullback_morphismRestrict f U).flip hf


instance (priority := 900) : P.RespectsIso := by
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    Q : AlgebraicGeometry.AffineTargetMorphismProperty
    inst✝ : AlgebraicGeometry.HasAffineProperty P Q
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    ⊢ P.RespectsIso
  -/
  letI := isLocal_affineProperty P
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    Q : AlgebraicGeometry.AffineTargetMorphismProperty
    inst✝ : AlgebraicGeometry.HasAffineProperty P Q
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    this : Q.IsLocal := AlgebraicGeometry.HasAffineProperty.isLocal_affineProperty P
    ⊢ P.RespectsIso
  -/
  rw [eq_targetAffineLocally P]
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    Q : AlgebraicGeometry.AffineTargetMorphismProperty
    inst✝ : AlgebraicGeometry.HasAffineProperty P Q
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    this : Q.IsLocal := AlgebraicGeometry.HasAffineProperty.isLocal_affineProperty P
    ⊢ (AlgebraicGeometry.targetAffineLocally Q).RespectsIso
  -/
  infer_instance
  /-
    🎉 no goals
  -/


theorem of_iSup_eq_top
    {ι} (U : ι → Y.affineOpens) (hU : ⨆ i, (U i : Y.Opens) = ⊤)
    (hU' : ∀ i, Q (f ∣_ U i)) :
    P f := by
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    Q : AlgebraicGeometry.AffineTargetMorphismProperty
    inst✝ : AlgebraicGeometry.HasAffineProperty P Q
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    ι : Sort u_1
    U : ι → ↑Y.affineOpens
    hU : Eq (iSup fun i => ↑(U i)) Top.top
    hU' : ∀ (i : ι), Q (AlgebraicGeometry.morphismRestrict f ↑(U i))
    ⊢ P f
  -/
  letI := isLocal_affineProperty P
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    Q : AlgebraicGeometry.AffineTargetMorphismProperty
    inst✝ : AlgebraicGeometry.HasAffineProperty P Q
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    ι : Sort u_1
    U : ι → ↑Y.affineOpens
    hU : Eq (iSup fun i => ↑(U i)) Top.top
    hU' : ∀ (i : ι), Q (AlgebraicGeometry.morphismRestrict f ↑(U i))
    this : Q.IsLocal := AlgebraicGeometry.HasAffineProperty.isLocal_affineProperty P
    ⊢ P f
  -/
  rw [eq_targetAffineLocally P]
  classical
  intro V
  induction V using of_affine_open_cover U hU  with
  | basicOpen U r h =>
    haveI : IsAffine _ := U.2
    have := AffineTargetMorphismProperty.IsLocal.to_basicOpen (f ∣_ U.1) (U.1.topIso.inv r) h
    exact (Q.arrow_mk_iso_iff
      (morphismRestrictRestrictBasicOpen f _ r)).mp this
  | openCover U s hs H =>
    apply AffineTargetMorphismProperty.IsLocal.of_basicOpenCover _
      (s.image (Scheme.Opens.topIso _).inv) (by simp [← Ideal.map_span, hs, Ideal.map_top])
    intro ⟨r, hr⟩
    obtain ⟨r, hr', rfl⟩ := Finset.mem_image.mp hr
    exact (Q.arrow_mk_iso_iff
      (morphismRestrictRestrictBasicOpen f _ r).symm).mp (H ⟨r, hr'⟩)
  | hU i => exact hU' i


theorem iff_of_iSup_eq_top
    {ι} (U : ι → Y.affineOpens) (hU : ⨆ i, (U i : Y.Opens) = ⊤) :
    P f ↔ ∀ i, Q (f ∣_ U i) :=
  ⟨fun H _ ↦ restrict H _, fun H ↦ HasAffineProperty.of_iSup_eq_top U hU H⟩


theorem of_openCover
    (𝒰 : Y.OpenCover) [∀ i, IsAffine (𝒰.obj i)] (h𝒰 : ∀ i, Q (𝒰.pullbackHom f i)) :
    P f :=
  letI := isLocal_affineProperty P
  of_iSup_eq_top
    (fun i ↦ ⟨_, isAffineOpen_opensRange (𝒰.map i)⟩) 𝒰.iSup_opensRange
    (fun i ↦ (Q.arrow_mk_iso_iff (morphismRestrictOpensRange f _)).mpr (h𝒰 i))


theorem iff_of_openCover (𝒰 : Y.OpenCover) [∀ i, IsAffine (𝒰.obj i)] :
    P f ↔ ∀ i, Q (𝒰.pullbackHom f i) := by
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    Q : AlgebraicGeometry.AffineTargetMorphismProperty
    inst✝¹ : AlgebraicGeometry.HasAffineProperty P Q
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    𝒰 : Y.OpenCover
    inst✝ : ∀ (i : 𝒰.J), AlgebraicGeometry.IsAffine (𝒰.obj i)
    ⊢ Iff (P f) (∀ (i : 𝒰.1), Q (AlgebraicGeometry.Scheme.Cover.pullbackHom 𝒰 f i))
  -/
  letI := isLocal_affineProperty P
  rw [iff_of_iSup_eq_top (P := P)
    (fun i ↦ ⟨_, isAffineOpen_opensRange _⟩) 𝒰.iSup_opensRange]
  exact forall_congr' fun i ↦ Q.arrow_mk_iso_iff
    (morphismRestrictOpensRange f _)


theorem iff_of_isAffine [IsAffine Y] : P f ↔ Q f := by
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    Q : AlgebraicGeometry.AffineTargetMorphismProperty
    inst✝¹ : AlgebraicGeometry.HasAffineProperty P Q
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝ : AlgebraicGeometry.IsAffine Y
    ⊢ Iff (P f) (Q f)
  -/
  letI := isLocal_affineProperty P
  haveI : ∀ i, IsAffine (Scheme.Cover.obj
      (Scheme.coverOfIsIso (P := @IsOpenImmersion) (𝟙 Y)) i) := fun i => by
    dsimp; infer_instance
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    Q : AlgebraicGeometry.AffineTargetMorphismProperty
    inst✝¹ : AlgebraicGeometry.HasAffineProperty P Q
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝ : AlgebraicGeometry.IsAffine Y
    this✝ : Q.IsLocal := AlgebraicGeometry.HasAffineProperty.isLocal_affinePropert …
    this : ∀ (i : (AlgebraicGeometry.Scheme.coverOfIsIso (CategoryTheory.CategoryS …
    ⊢ Iff (P f) (Q f)
  -/
  rw [iff_of_openCover (P := P) (Scheme.coverOfIsIso.{0} (𝟙 Y))]
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    Q : AlgebraicGeometry.AffineTargetMorphismProperty
    inst✝¹ : AlgebraicGeometry.HasAffineProperty P Q
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝ : AlgebraicGeometry.IsAffine Y
    this✝ : Q.IsLocal := AlgebraicGeometry.HasAffineProperty.isLocal_affinePropert …
    this : ∀ (i : (AlgebraicGeometry.Scheme.coverOfIsIso (CategoryTheory.CategoryS …
    ⊢ Iff (∀ (i : (AlgebraicGeometry.Scheme.coverOfIsIso (CategoryTheory.CategoryS …
  -/
  trans Q (pullback.snd f (𝟙 _))
    /-
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      Q : AlgebraicGeometry.AffineTargetMorphismProperty
      inst✝¹ : AlgebraicGeometry.HasAffineProperty P Q
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      inst✝ : AlgebraicGeometry.IsAffine Y
      this✝ : Q.IsLocal := AlgebraicGeometry.HasAffineProperty.isLocal_affinePropert …
      this : ∀ (i : (AlgebraicGeometry.Scheme.coverOfIsIso (CategoryTheory.CategoryS …
      ⊢ Iff (∀ (i : (AlgebraicGeometry.Scheme.coverOfIsIso (CategoryTheory.CategoryS …
    -/
  · exact ⟨fun H => H PUnit.unit, fun H _ => H⟩
    /-
      🎉 no goals
    -/
  rw [← Category.comp_id (pullback.snd _ _), ← pullback.condition,
    Q.cancel_left_of_respectsIso]


instance (priority := 900) : IsLocalAtTarget P := by
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    Q : AlgebraicGeometry.AffineTargetMorphismProperty
    inst✝ : AlgebraicGeometry.HasAffineProperty P Q
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    ⊢ AlgebraicGeometry.IsLocalAtTarget P
  -/
  letI := isLocal_affineProperty P
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    Q : AlgebraicGeometry.AffineTargetMorphismProperty
    inst✝ : AlgebraicGeometry.HasAffineProperty P Q
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    this : Q.IsLocal := AlgebraicGeometry.HasAffineProperty.isLocal_affineProperty P
    ⊢ AlgebraicGeometry.IsLocalAtTarget P
  -/
  apply IsLocalAtTarget.mk'
    /-
      case restrict
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      Q : AlgebraicGeometry.AffineTargetMorphismProperty
      inst✝ : AlgebraicGeometry.HasAffineProperty P Q
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      this : Q.IsLocal := AlgebraicGeometry.HasAffineProperty.isLocal_affineProperty P
      ⊢ ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y) (U : Y.Opens), P f → …
    -/
  · rw [eq_targetAffineLocally P]
    /-
      case restrict
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      Q : AlgebraicGeometry.AffineTargetMorphismProperty
      inst✝ : AlgebraicGeometry.HasAffineProperty P Q
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      this : Q.IsLocal := AlgebraicGeometry.HasAffineProperty.isLocal_affineProperty P
      ⊢ ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y) (U : Y.Opens), Algeb …
    -/
    intro X Y f U H V
    /-
      case restrict
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      Q : AlgebraicGeometry.AffineTargetMorphismProperty
      inst✝ : AlgebraicGeometry.HasAffineProperty P Q
      X✝ Y✝ : AlgebraicGeometry.Scheme
      f✝ : Quiver.Hom X✝ Y✝
      this : Q.IsLocal := AlgebraicGeometry.HasAffineProperty.isLocal_affineProperty P
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      U : Y.Opens
      H : AlgebraicGeometry.targetAffineLocally Q f
      V : ↑(↑U).affineOpens
      ⊢ Q (AlgebraicGeometry.morphismRestrict (AlgebraicGeometry.morphismRestrict f  …
    -/
    rw [Q.arrow_mk_iso_iff (morphismRestrictRestrict f _ _)]
    /-
      case restrict
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      Q : AlgebraicGeometry.AffineTargetMorphismProperty
      inst✝ : AlgebraicGeometry.HasAffineProperty P Q
      X✝ Y✝ : AlgebraicGeometry.Scheme
      f✝ : Quiver.Hom X✝ Y✝
      this : Q.IsLocal := AlgebraicGeometry.HasAffineProperty.isLocal_affineProperty P
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      U : Y.Opens
      H : AlgebraicGeometry.targetAffineLocally Q f
      V : ↑(↑U).affineOpens
      ⊢ Q (AlgebraicGeometry.morphismRestrict f ((AlgebraicGeometry.Scheme.Hom.opens …
    -/
    exact H ⟨_, V.2.image_of_isOpenImmersion (Y.ofRestrict _)⟩
    /-
      🎉 no goals
    -/
    /-
      case of_sSup_eq_top
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      Q : AlgebraicGeometry.AffineTargetMorphismProperty
      inst✝ : AlgebraicGeometry.HasAffineProperty P Q
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      this : Q.IsLocal := AlgebraicGeometry.HasAffineProperty.isLocal_affineProperty P
      ⊢ ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y) {ι : Type u_1} (U :  …
    -/
  · rintro X Y f ι U hU H
    /-
      case of_sSup_eq_top
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      Q : AlgebraicGeometry.AffineTargetMorphismProperty
      inst✝ : AlgebraicGeometry.HasAffineProperty P Q
      X✝ Y✝ : AlgebraicGeometry.Scheme
      f✝ : Quiver.Hom X✝ Y✝
      this : Q.IsLocal := AlgebraicGeometry.HasAffineProperty.isLocal_affineProperty P
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      ι : Type u_1
      U : ι → Y.Opens
      hU : Eq (iSup U) Top.top
      H : ∀ (i : ι), P (AlgebraicGeometry.morphismRestrict f (U i))
      ⊢ P f
    -/
    let 𝒰 := Y.openCoverOfISupEqTop U hU
    /-
      case of_sSup_eq_top
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      Q : AlgebraicGeometry.AffineTargetMorphismProperty
      inst✝ : AlgebraicGeometry.HasAffineProperty P Q
      X✝ Y✝ : AlgebraicGeometry.Scheme
      f✝ : Quiver.Hom X✝ Y✝
      this : Q.IsLocal := AlgebraicGeometry.HasAffineProperty.isLocal_affineProperty P
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      ι : Type u_1
      U : ι → Y.Opens
      hU : Eq (iSup U) Top.top
      H : ∀ (i : ι), P (AlgebraicGeometry.morphismRestrict f (U i))
      𝒰 : Y.OpenCover := Y.openCoverOfISupEqTop U hU
      ⊢ P f
    -/
    apply of_openCover 𝒰.affineRefinement.openCover
    /-
      case of_sSup_eq_top
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      Q : AlgebraicGeometry.AffineTargetMorphismProperty
      inst✝ : AlgebraicGeometry.HasAffineProperty P Q
      X✝ Y✝ : AlgebraicGeometry.Scheme
      f✝ : Quiver.Hom X✝ Y✝
      this : Q.IsLocal := AlgebraicGeometry.HasAffineProperty.isLocal_affineProperty P
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      ι : Type u_1
      U : ι → Y.Opens
      hU : Eq (iSup U) Top.top
      H : ∀ (i : ι), P (AlgebraicGeometry.morphismRestrict f (U i))
      𝒰 : Y.OpenCover := Y.openCoverOfISupEqTop U hU
      ⊢ ∀ (i : 𝒰.affineRefinement.openCover.1), Q (AlgebraicGeometry.Scheme.Cover.pu …
    -/
    rintro ⟨i, j⟩
    have : P (𝒰.pullbackHom f i) := by
      refine (P.arrow_mk_iso_iff
        (morphismRestrictEq _ ?_ ≪≫ morphismRestrictOpensRange f (𝒰.map i))).mp (H i)
      exact (Scheme.Opens.opensRange_ι _).symm
    rw [← Q.cancel_left_of_respectsIso (𝒰.pullbackCoverAffineRefinementObjIso f _).inv,
      𝒰.pullbackCoverAffineRefinementObjIso_inv_pullbackHom]
    /-
      case of_sSup_eq_top.mk
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      Q : AlgebraicGeometry.AffineTargetMorphismProperty
      inst✝ : AlgebraicGeometry.HasAffineProperty P Q
      X✝ Y✝ : AlgebraicGeometry.Scheme
      f✝ : Quiver.Hom X✝ Y✝
      this✝ : Q.IsLocal := AlgebraicGeometry.HasAffineProperty.isLocal_affinePropert …
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      ι : Type u_1
      U : ι → Y.Opens
      hU : Eq (iSup U) Top.top
      H : ∀ (i : ι), P (AlgebraicGeometry.morphismRestrict f (U i))
      𝒰 : Y.OpenCover := Y.openCoverOfISupEqTop U hU
      i : 𝒰.J
      j : ((fun j => (𝒰.obj j).affineCover) i).J
      this : P (AlgebraicGeometry.Scheme.Cover.pullbackHom 𝒰 f i)
      ⊢ Q (AlgebraicGeometry.Scheme.Cover.pullbackHom (𝒰.obj ⟨i, j⟩.fst).affineCover …
    -/
    exact of_isPullback (.of_hasPullback _ _) this
    /-
      🎉 no goals
    -/


open AffineTargetMorphismProperty in
protected theorem iff {P : MorphismProperty Scheme} {Q : AffineTargetMorphismProperty} :
    HasAffineProperty P Q ↔ IsLocalAtTarget P ∧ Q = of P :=
  ⟨fun _ ↦ ⟨inferInstance, ext fun _ _ _ ↦ iff_of_isAffine.symm⟩,
    fun ⟨_, e⟩ ↦ e ▸ of_isLocalAtTarget P⟩


private theorem pullback_fst_of_right (hP' : Q.IsStableUnderBaseChange)
    {X Y S : Scheme} (f : X ⟶ S) (g : Y ⟶ S) [IsAffine S] (H : Q g) :
    P (pullback.fst f g) := by
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    Q : AlgebraicGeometry.AffineTargetMorphismProperty
    inst✝¹ : AlgebraicGeometry.HasAffineProperty P Q
    hP' : Q.IsStableUnderBaseChange
    X Y S : AlgebraicGeometry.Scheme
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    inst✝ : AlgebraicGeometry.IsAffine S
    H : Q g
    ⊢ P (CategoryTheory.Limits.pullback.fst f g)
  -/
  letI := isLocal_affineProperty P
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    Q : AlgebraicGeometry.AffineTargetMorphismProperty
    inst✝¹ : AlgebraicGeometry.HasAffineProperty P Q
    hP' : Q.IsStableUnderBaseChange
    X Y S : AlgebraicGeometry.Scheme
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    inst✝ : AlgebraicGeometry.IsAffine S
    H : Q g
    this : Q.IsLocal := AlgebraicGeometry.HasAffineProperty.isLocal_affineProperty P
    ⊢ P (CategoryTheory.Limits.pullback.fst f g)
  -/
  rw [iff_of_openCover (P := P) X.affineCover]
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    Q : AlgebraicGeometry.AffineTargetMorphismProperty
    inst✝¹ : AlgebraicGeometry.HasAffineProperty P Q
    hP' : Q.IsStableUnderBaseChange
    X Y S : AlgebraicGeometry.Scheme
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    inst✝ : AlgebraicGeometry.IsAffine S
    H : Q g
    this : Q.IsLocal := AlgebraicGeometry.HasAffineProperty.isLocal_affineProperty P
    ⊢ ∀ (i : X.affineCover.1), Q (AlgebraicGeometry.Scheme.Cover.pullbackHom X.aff …
  -/
  intro i
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    Q : AlgebraicGeometry.AffineTargetMorphismProperty
    inst✝¹ : AlgebraicGeometry.HasAffineProperty P Q
    hP' : Q.IsStableUnderBaseChange
    X Y S : AlgebraicGeometry.Scheme
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    inst✝ : AlgebraicGeometry.IsAffine S
    H : Q g
    this : Q.IsLocal := AlgebraicGeometry.HasAffineProperty.isLocal_affineProperty P
    i : X.affineCover.1
    ⊢ Q (AlgebraicGeometry.Scheme.Cover.pullbackHom X.affineCover (CategoryTheory. …
  -/
  let e := pullbackSymmetry _ _ ≪≫ pullbackRightPullbackFstIso f g (X.affineCover.map i)
  have : e.hom ≫ pullback.fst _ _ = X.affineCover.pullbackHom (pullback.fst _ _) i := by
    simp [e, Scheme.Cover.pullbackHom]
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    Q : AlgebraicGeometry.AffineTargetMorphismProperty
    inst✝¹ : AlgebraicGeometry.HasAffineProperty P Q
    hP' : Q.IsStableUnderBaseChange
    X Y S : AlgebraicGeometry.Scheme
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    inst✝ : AlgebraicGeometry.IsAffine S
    H : Q g
    this✝ : Q.IsLocal := AlgebraicGeometry.HasAffineProperty.isLocal_affinePropert …
    i : X.affineCover.1
    e : CategoryTheory.Iso (CategoryTheory.Limits.pullback (CategoryTheory.Limits. …
    this : Eq (CategoryTheory.CategoryStruct.comp e.hom (CategoryTheory.Limits.pul …
    ⊢ Q (AlgebraicGeometry.Scheme.Cover.pullbackHom X.affineCover (CategoryTheory. …
  -/
  rw [← this, Q.cancel_left_of_respectsIso]
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    Q : AlgebraicGeometry.AffineTargetMorphismProperty
    inst✝¹ : AlgebraicGeometry.HasAffineProperty P Q
    hP' : Q.IsStableUnderBaseChange
    X Y S : AlgebraicGeometry.Scheme
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    inst✝ : AlgebraicGeometry.IsAffine S
    H : Q g
    this✝ : Q.IsLocal := AlgebraicGeometry.HasAffineProperty.isLocal_affinePropert …
    i : X.affineCover.1
    e : CategoryTheory.Iso (CategoryTheory.Limits.pullback (CategoryTheory.Limits. …
    this : Eq (CategoryTheory.CategoryStruct.comp e.hom (CategoryTheory.Limits.pul …
    ⊢ Q (CategoryTheory.Limits.pullback.fst (CategoryTheory.CategoryStruct.comp (X …
  -/
  apply hP' (.of_hasPullback _ _)
  /-
    P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
    Q : AlgebraicGeometry.AffineTargetMorphismProperty
    inst✝¹ : AlgebraicGeometry.HasAffineProperty P Q
    hP' : Q.IsStableUnderBaseChange
    X Y S : AlgebraicGeometry.Scheme
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    inst✝ : AlgebraicGeometry.IsAffine S
    H : Q g
    this✝ : Q.IsLocal := AlgebraicGeometry.HasAffineProperty.isLocal_affinePropert …
    i : X.affineCover.1
    e : CategoryTheory.Iso (CategoryTheory.Limits.pullback (CategoryTheory.Limits. …
    this : Eq (CategoryTheory.CategoryStruct.comp e.hom (CategoryTheory.Limits.pul …
    ⊢ Q g
  -/
  exact H
  /-
    🎉 no goals
  -/


theorem isStableUnderBaseChange (hP' : Q.IsStableUnderBaseChange) :
    P.IsStableUnderBaseChange :=
  MorphismProperty.IsStableUnderBaseChange.mk'
    (fun X Y S f g _ H => by
      /-
        P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
        Q : AlgebraicGeometry.AffineTargetMorphismProperty
        inst✝ : AlgebraicGeometry.HasAffineProperty P Q
        hP' : Q.IsStableUnderBaseChange
        X Y S : AlgebraicGeometry.Scheme
        f : Quiver.Hom X S
        g : Quiver.Hom Y S
        x✝ : CategoryTheory.Limits.HasPullback f g
        H : P g
        ⊢ P (CategoryTheory.Limits.pullback.fst f g)
      -/
      rw [IsLocalAtTarget.iff_of_openCover (P := P) (S.affineCover.pullbackCover f)]
      /-
        P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
        Q : AlgebraicGeometry.AffineTargetMorphismProperty
        inst✝ : AlgebraicGeometry.HasAffineProperty P Q
        hP' : Q.IsStableUnderBaseChange
        X Y S : AlgebraicGeometry.Scheme
        f : Quiver.Hom X S
        g : Quiver.Hom Y S
        x✝ : CategoryTheory.Limits.HasPullback f g
        H : P g
        ⊢ ∀ (i : (AlgebraicGeometry.Scheme.Cover.pullbackCover S.affineCover f).1), P  …
      -/
      intro i
      let e : pullback (pullback.fst f g) ((S.affineCover.pullbackCover f).map i) ≅
          _ := by
        refine pullbackSymmetry _ _ ≪≫ pullbackRightPullbackFstIso f g _ ≪≫ ?_ ≪≫
          (pullbackRightPullbackFstIso (S.affineCover.map i) g
            (pullback.snd f (S.affineCover.map i))).symm
        exact asIso
          (pullback.map _ _ _ _ (𝟙 _) (𝟙 _) (𝟙 _) (by simpa using pullback.condition) (by simp))
      have : e.hom ≫ pullback.fst _ _ =
          (S.affineCover.pullbackCover f).pullbackHom (pullback.fst _ _) i := by
        simp [e, Scheme.Cover.pullbackHom]
      /-
        P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
        Q : AlgebraicGeometry.AffineTargetMorphismProperty
        inst✝ : AlgebraicGeometry.HasAffineProperty P Q
        hP' : Q.IsStableUnderBaseChange
        X Y S : AlgebraicGeometry.Scheme
        f : Quiver.Hom X S
        g : Quiver.Hom Y S
        x✝ : CategoryTheory.Limits.HasPullback f g
        H : P g
        i : (AlgebraicGeometry.Scheme.Cover.pullbackCover S.affineCover f).1
        e : CategoryTheory.Iso (CategoryTheory.Limits.pullback (CategoryTheory.Limits. …
        this : Eq (CategoryTheory.CategoryStruct.comp e.hom (CategoryTheory.Limits.pul …
        ⊢ P ((AlgebraicGeometry.Scheme.Cover.pullbackCover S.affineCover f).pullbackHo …
      -/
      rw [← this, P.cancel_left_of_respectsIso]
      /-
        P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
        Q : AlgebraicGeometry.AffineTargetMorphismProperty
        inst✝ : AlgebraicGeometry.HasAffineProperty P Q
        hP' : Q.IsStableUnderBaseChange
        X Y S : AlgebraicGeometry.Scheme
        f : Quiver.Hom X S
        g : Quiver.Hom Y S
        x✝ : CategoryTheory.Limits.HasPullback f g
        H : P g
        i : (AlgebraicGeometry.Scheme.Cover.pullbackCover S.affineCover f).1
        e : CategoryTheory.Iso (CategoryTheory.Limits.pullback (CategoryTheory.Limits. …
        this : Eq (CategoryTheory.CategoryStruct.comp e.hom (CategoryTheory.Limits.pul …
        ⊢ P (CategoryTheory.Limits.pullback.fst (CategoryTheory.Limits.pullback.snd f  …
      -/
      apply HasAffineProperty.pullback_fst_of_right hP'
      /-
        case H
        P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
        Q : AlgebraicGeometry.AffineTargetMorphismProperty
        inst✝ : AlgebraicGeometry.HasAffineProperty P Q
        hP' : Q.IsStableUnderBaseChange
        X Y S : AlgebraicGeometry.Scheme
        f : Quiver.Hom X S
        g : Quiver.Hom Y S
        x✝ : CategoryTheory.Limits.HasPullback f g
        H : P g
        i : (AlgebraicGeometry.Scheme.Cover.pullbackCover S.affineCover f).1
        e : CategoryTheory.Iso (CategoryTheory.Limits.pullback (CategoryTheory.Limits. …
        this : Eq (CategoryTheory.CategoryStruct.comp e.hom (CategoryTheory.Limits.pul …
        ⊢ Q (CategoryTheory.Limits.pullback.fst (S.affineCover.map i) g)
      -/
      letI := isLocal_affineProperty P
      /-
        case H
        P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
        Q : AlgebraicGeometry.AffineTargetMorphismProperty
        inst✝ : AlgebraicGeometry.HasAffineProperty P Q
        hP' : Q.IsStableUnderBaseChange
        X Y S : AlgebraicGeometry.Scheme
        f : Quiver.Hom X S
        g : Quiver.Hom Y S
        x✝ : CategoryTheory.Limits.HasPullback f g
        H : P g
        i : (AlgebraicGeometry.Scheme.Cover.pullbackCover S.affineCover f).1
        e : CategoryTheory.Iso (CategoryTheory.Limits.pullback (CategoryTheory.Limits. …
        this✝ : Eq (CategoryTheory.CategoryStruct.comp e.hom (CategoryTheory.Limits.pu …
        this : Q.IsLocal := AlgebraicGeometry.HasAffineProperty.isLocal_affineProperty P
        ⊢ Q (CategoryTheory.Limits.pullback.fst (S.affineCover.map i) g)
      -/
      rw [← pullbackSymmetry_hom_comp_snd, Q.cancel_left_of_respectsIso]
      /-
        case H
        P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
        Q : AlgebraicGeometry.AffineTargetMorphismProperty
        inst✝ : AlgebraicGeometry.HasAffineProperty P Q
        hP' : Q.IsStableUnderBaseChange
        X Y S : AlgebraicGeometry.Scheme
        f : Quiver.Hom X S
        g : Quiver.Hom Y S
        x✝ : CategoryTheory.Limits.HasPullback f g
        H : P g
        i : (AlgebraicGeometry.Scheme.Cover.pullbackCover S.affineCover f).1
        e : CategoryTheory.Iso (CategoryTheory.Limits.pullback (CategoryTheory.Limits. …
        this✝ : Eq (CategoryTheory.CategoryStruct.comp e.hom (CategoryTheory.Limits.pu …
        this : Q.IsLocal := AlgebraicGeometry.HasAffineProperty.isLocal_affineProperty P
        ⊢ Q (CategoryTheory.Limits.pullback.snd g (S.affineCover.map i))
      -/
      apply of_isPullback (.of_hasPullback _ _) H)
      /-
        🎉 no goals
      -/


lemma isLocalAtSource
    (H : ∀ {X Y : Scheme.{u}} (f : X ⟶ Y) [IsAffine Y] (𝒰 : Scheme.OpenCover.{u} X),
        Q f ↔ ∀ i, Q (𝒰.map i ≫ f)) : IsLocalAtSource P where
  iff_of_openCover' {X Y} f 𝒰 := by
    /-
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      Q : AlgebraicGeometry.AffineTargetMorphismProperty
      inst✝ : AlgebraicGeometry.HasAffineProperty P Q
      H : ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y) [inst : AlgebraicG …
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      𝒰 : X.OpenCover
      ⊢ Iff (P f) (∀ (i : 𝒰.J), P (CategoryTheory.CategoryStruct.comp (𝒰.map i) f))
    -/
    simp_rw [IsLocalAtTarget.iff_of_iSup_eq_top _ (iSup_affineOpens_eq_top Y)]
    /-
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      Q : AlgebraicGeometry.AffineTargetMorphismProperty
      inst✝ : AlgebraicGeometry.HasAffineProperty P Q
      H : ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y) [inst : AlgebraicG …
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      𝒰 : X.OpenCover
      ⊢ Iff (∀ (i : ↑Y.affineOpens), P (AlgebraicGeometry.morphismRestrict f ↑i)) (∀ …
    -/
    rw [forall_comm]
    /-
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      Q : AlgebraicGeometry.AffineTargetMorphismProperty
      inst✝ : AlgebraicGeometry.HasAffineProperty P Q
      H : ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y) [inst : AlgebraicG …
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      𝒰 : X.OpenCover
      ⊢ Iff (∀ (i : ↑Y.affineOpens), P (AlgebraicGeometry.morphismRestrict f ↑i)) (∀ …
    -/
    refine forall_congr' fun U ↦ ?_
    /-
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      Q : AlgebraicGeometry.AffineTargetMorphismProperty
      inst✝ : AlgebraicGeometry.HasAffineProperty P Q
      H : ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y) [inst : AlgebraicG …
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      𝒰 : X.OpenCover
      U : ↑Y.affineOpens
      ⊢ Iff (P (AlgebraicGeometry.morphismRestrict f ↑U)) (∀ (a : 𝒰.J), P (Algebraic …
    -/
    simp_rw [HasAffineProperty.iff_of_isAffine, morphismRestrict_comp]
    /-
      P : CategoryTheory.MorphismProperty AlgebraicGeometry.Scheme
      Q : AlgebraicGeometry.AffineTargetMorphismProperty
      inst✝ : AlgebraicGeometry.HasAffineProperty P Q
      H : ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y) [inst : AlgebraicG …
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      𝒰 : X.OpenCover
      U : ↑Y.affineOpens
      ⊢ Iff (Q (AlgebraicGeometry.morphismRestrict f ↑U)) (∀ (a : 𝒰.J), Q (CategoryT …
    -/
    exact @H _ _ (f ∣_ U.1) U.2 (𝒰.restrict (f ⁻¹ᵁ U.1))
    /-
      🎉 no goals
    -/


