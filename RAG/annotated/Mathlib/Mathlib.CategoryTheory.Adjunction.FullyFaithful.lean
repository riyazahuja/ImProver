/-- If the left adjoint is faithful, then each component of the unit is an monomorphism. -/
instance unit_mono_of_L_faithful [L.Faithful] (X : C) : Mono (h.unit.app X) where
  right_cancellation {Y} f g hfg :=
                                                                /-
                                                                  C : Type u₁
                                                                  inst✝² : CategoryTheory.Category.{v₁, u₁} C
                                                                  D : Type u₂
                                                                  inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                                                                  L : CategoryTheory.Functor C D
                                                                  R : CategoryTheory.Functor D C
                                                                  h : CategoryTheory.Adjunction L R
                                                                  inst✝ : L.Faithful
                                                                  X Y : C
                                                                  f g : Quiver.Hom Y ((CategoryTheory.Functor.id C).obj X)
                                                                  hfg : Eq (CategoryTheory.CategoryStruct.comp f (h.unit.app X)) (CategoryTheory …
                                                                  ⊢ Eq ((h.homEquiv Y (L.obj X)) (L.map f)) ((h.homEquiv Y (L.obj X)) (L.map g))
                                                                -/
    L.map_injective <| (h.homEquiv Y (L.obj X)).injective <| by simpa using hfg
                                                                /-
                                                                  🎉 no goals
                                                                -/


/-- If the left adjoint is full, then each component of the unit is a split epimorphism.-/
noncomputable def unitSplitEpiOfLFull [L.Full] (X : C) : SplitEpi (h.unit.app X) where
  section_ := L.preimage (h.counit.app (L.obj X))
           /-
             C : Type u₁
             inst✝² : CategoryTheory.Category.{v₁, u₁} C
             D : Type u₂
             inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
             L : CategoryTheory.Functor C D
             R : CategoryTheory.Functor D C
             h : CategoryTheory.Adjunction L R
             inst✝ : L.Full
             X : C
             ⊢ Eq (CategoryTheory.CategoryStruct.comp (L.preimage (h.counit.app (L.obj X))) …
           -/
  id := by simp [← h.unit_naturality (L.preimage (h.counit.app (L.obj X)))]
           /-
             🎉 no goals
           -/


/-- If the right adjoint is full, then each component of the counit is a split monomorphism. -/
instance unit_isSplitEpi_of_L_full [L.Full] (X : C) : IsSplitEpi (h.unit.app X) :=
  ⟨⟨h.unitSplitEpiOfLFull X⟩⟩


instance [L.Full] [L.Faithful] (X : C) : IsIso (h.unit.app X) :=
  isIso_of_mono_of_isSplitEpi _


/-- If the left adjoint is fully faithful, then the unit is an isomorphism. -/
instance unit_isIso_of_L_fully_faithful [L.Full] [L.Faithful] : IsIso (Adjunction.unit h) :=
  NatIso.isIso_of_isIso_app _


/-- If the right adjoint is faithful, then each component of the counit is an epimorphism.-/
instance counit_epi_of_R_faithful [R.Faithful] (X : D) : Epi (h.counit.app X) where
  left_cancellation {Y} f g hfg :=
                                                                     /-
                                                                       C : Type u₁
                                                                       inst✝² : CategoryTheory.Category.{v₁, u₁} C
                                                                       D : Type u₂
                                                                       inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                                                                       L : CategoryTheory.Functor C D
                                                                       R : CategoryTheory.Functor D C
                                                                       h : CategoryTheory.Adjunction L R
                                                                       inst✝ : R.Faithful
                                                                       X Y : D
                                                                       f g : Quiver.Hom ((CategoryTheory.Functor.id D).obj X) Y
                                                                       hfg : Eq (CategoryTheory.CategoryStruct.comp (h.counit.app X) f) (CategoryTheo …
                                                                       ⊢ Eq ((h.homEquiv (R.obj X) Y).symm (R.map f)) ((h.homEquiv (R.obj X) Y).symm  …
                                                                     -/
    R.map_injective <| (h.homEquiv (R.obj X) Y).symm.injective <| by simpa using hfg
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


/-- If the right adjoint is full, then each component of the counit is a split monomorphism. -/
noncomputable def counitSplitMonoOfRFull [R.Full] (X : D) : SplitMono (h.counit.app X) where
  retraction := R.preimage (h.unit.app (R.obj X))
           /-
             C : Type u₁
             inst✝² : CategoryTheory.Category.{v₁, u₁} C
             D : Type u₂
             inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
             L : CategoryTheory.Functor C D
             R : CategoryTheory.Functor D C
             h : CategoryTheory.Adjunction L R
             inst✝ : R.Full
             X : D
             ⊢ Eq (CategoryTheory.CategoryStruct.comp (h.counit.app X) (R.preimage (h.unit. …
           -/
  id := by simp [← h.counit_naturality (R.preimage (h.unit.app (R.obj X)))]
           /-
             🎉 no goals
           -/


/-- If the right adjoint is full, then each component of the counit is a split monomorphism. -/
instance counit_isSplitMono_of_R_full [R.Full] (X : D) : IsSplitMono (h.counit.app X) :=
  ⟨⟨h.counitSplitMonoOfRFull X⟩⟩


instance [R.Full] [R.Faithful] (X : D) : IsIso (h.counit.app X) :=
  isIso_of_epi_of_isSplitMono _


/-- If the right adjoint is fully faithful, then the counit is an isomorphism. -/
instance counit_isIso_of_R_fully_faithful [R.Full] [R.Faithful] : IsIso (Adjunction.counit h) :=
  NatIso.isIso_of_isIso_app _


/-- If the unit of an adjunction is an isomorphism, then its inverse on the image of L is given
by L whiskered with the counit. -/
@[simp]
theorem inv_map_unit {X : C} [IsIso (h.unit.app X)] :
    inv (L.map (h.unit.app X)) = h.counit.app (L.obj X) :=
  IsIso.inv_eq_of_hom_inv_id (h.left_triangle_components X)


/-- If the unit is an isomorphism, bundle one has an isomorphism `L ⋙ R ⋙ L ≅ L`. -/
@[simps!]
noncomputable def whiskerLeftLCounitIsoOfIsIsoUnit [IsIso h.unit] : L ⋙ R ⋙ L ≅ L :=
  (L.associator R L).symm ≪≫ isoWhiskerRight (asIso h.unit).symm L ≪≫ Functor.leftUnitor _


/-- If the counit of an adjunction is an isomorphism, then its inverse on the image of R is given
by R whiskered with the unit. -/
@[simp]
theorem inv_counit_map {X : D} [IsIso (h.counit.app X)] :
    inv (R.map (h.counit.app X)) = h.unit.app (R.obj X) :=
  IsIso.inv_eq_of_inv_hom_id (h.right_triangle_components X)


/-- If the counit of an is an isomorphism, one has an isomorphism `(R ⋙ L ⋙ R) ≅ R`. -/
@[simps!]
noncomputable def whiskerLeftRUnitIsoOfIsIsoCounit [IsIso h.counit] : R ⋙ L ⋙ R ≅ R :=
  (R.associator L R).symm ≪≫ isoWhiskerRight (asIso h.counit) R ≪≫ Functor.leftUnitor _


/-- If each component of the unit is a monomorphism, then the left adjoint is faithful. -/
lemma faithful_L_of_mono_unit_app [∀ X, Mono (h.unit.app X)] : L.Faithful where
  map_injective {X Y f g} hfg := by
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      L : CategoryTheory.Functor C D
      R : CategoryTheory.Functor D C
      h : CategoryTheory.Adjunction L R
      inst✝ : ∀ (X : C), CategoryTheory.Mono (h.unit.app X)
      X Y : C
      f g : Quiver.Hom X Y
      hfg : Eq (L.map f) (L.map g)
      ⊢ Eq f g
    -/
    apply Mono.right_cancellation (f := h.unit.app Y)
    /-
      case a
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      L : CategoryTheory.Functor C D
      R : CategoryTheory.Functor D C
      h : CategoryTheory.Adjunction L R
      inst✝ : ∀ (X : C), CategoryTheory.Mono (h.unit.app X)
      X Y : C
      f g : Quiver.Hom X Y
      hfg : Eq (L.map f) (L.map g)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp f (h.unit.app Y)) (CategoryTheory.Cat …
    -/
    apply (h.homEquiv X (L.obj Y)).symm.injective
    /-
      case a.a
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      L : CategoryTheory.Functor C D
      R : CategoryTheory.Functor D C
      h : CategoryTheory.Adjunction L R
      inst✝ : ∀ (X : C), CategoryTheory.Mono (h.unit.app X)
      X Y : C
      f g : Quiver.Hom X Y
      hfg : Eq (L.map f) (L.map g)
      ⊢ Eq ((h.homEquiv X (L.obj Y)).symm (CategoryTheory.CategoryStruct.comp f (h.u …
    -/
    simpa using hfg
    /-
      🎉 no goals
    -/


/-- If each component of the unit is a split epimorphism, then the left adjoint is full. -/
lemma full_L_of_isSplitEpi_unit_app [∀ X, IsSplitEpi (h.unit.app X)] : L.Full where
  map_surjective {X Y} f := by
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      L : CategoryTheory.Functor C D
      R : CategoryTheory.Functor D C
      h : CategoryTheory.Adjunction L R
      inst✝ : ∀ (X : C), CategoryTheory.IsSplitEpi (h.unit.app X)
      X Y : C
      f : Quiver.Hom (L.obj X) (L.obj Y)
      ⊢ Exists fun a => Eq (L.map a) f
    -/
    use ((h.homEquiv X (L.obj Y)) f ≫ section_ (h.unit.app Y))
    /-
      case h
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      L : CategoryTheory.Functor C D
      R : CategoryTheory.Functor D C
      h : CategoryTheory.Adjunction L R
      inst✝ : ∀ (X : C), CategoryTheory.IsSplitEpi (h.unit.app X)
      X Y : C
      f : Quiver.Hom (L.obj X) (L.obj Y)
      ⊢ Eq (L.map (CategoryTheory.CategoryStruct.comp ((h.homEquiv X (L.obj Y)) f) ( …
    -/
    suffices L.map (section_ (h.unit.app Y)) = h.counit.app (L.obj Y) by simp [this]
    /-
      case h
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      L : CategoryTheory.Functor C D
      R : CategoryTheory.Functor D C
      h : CategoryTheory.Adjunction L R
      inst✝ : ∀ (X : C), CategoryTheory.IsSplitEpi (h.unit.app X)
      X Y : C
      f : Quiver.Hom (L.obj X) (L.obj Y)
      ⊢ Eq (L.map (CategoryTheory.section_ (h.unit.app Y))) (h.counit.app (L.obj Y))
    -/
    rw [← comp_id (L.map (section_ (h.unit.app Y)))]
    simp only [Functor.comp_obj, Functor.id_obj, comp_id, ← h.left_triangle_components Y,
      ← assoc, ← Functor.map_comp, IsSplitEpi.id, Functor.map_id, id_comp]


/-- If the unit is an isomorphism, then the left adjoint is fully faithful. -/
noncomputable def fullyFaithfulLOfIsIsoUnit [IsIso h.unit] : L.FullyFaithful where
  preimage {_ Y} f := h.homEquiv _ (L.obj Y) f ≫ inv (h.unit.app Y)


/-- If each component of the counit is an epimorphism, then the right adjoint is faithful. -/
lemma faithful_R_of_epi_counit_app [∀ X, Epi (h.counit.app X)] : R.Faithful where
  map_injective {X Y f g} hfg := by
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      L : CategoryTheory.Functor C D
      R : CategoryTheory.Functor D C
      h : CategoryTheory.Adjunction L R
      inst✝ : ∀ (X : D), CategoryTheory.Epi (h.counit.app X)
      X Y : D
      f g : Quiver.Hom X Y
      hfg : Eq (R.map f) (R.map g)
      ⊢ Eq f g
    -/
    apply Epi.left_cancellation (f := h.counit.app X)
    /-
      case a
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      L : CategoryTheory.Functor C D
      R : CategoryTheory.Functor D C
      h : CategoryTheory.Adjunction L R
      inst✝ : ∀ (X : D), CategoryTheory.Epi (h.counit.app X)
      X Y : D
      f g : Quiver.Hom X Y
      hfg : Eq (R.map f) (R.map g)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (h.counit.app X) f) (CategoryTheory.C …
    -/
    apply (h.homEquiv (R.obj X) Y).injective
    /-
      case a.a
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      L : CategoryTheory.Functor C D
      R : CategoryTheory.Functor D C
      h : CategoryTheory.Adjunction L R
      inst✝ : ∀ (X : D), CategoryTheory.Epi (h.counit.app X)
      X Y : D
      f g : Quiver.Hom X Y
      hfg : Eq (R.map f) (R.map g)
      ⊢ Eq ((h.homEquiv (R.obj X) Y) (CategoryTheory.CategoryStruct.comp (h.counit.a …
    -/
    simpa using hfg
    /-
      🎉 no goals
    -/


/-- If each component of the counit is a split monomorphism, then the right adjoint is full. -/
lemma full_R_of_isSplitMono_counit_app [∀ X, IsSplitMono (h.counit.app X)] : R.Full where
  map_surjective {X Y} f := by
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      L : CategoryTheory.Functor C D
      R : CategoryTheory.Functor D C
      h : CategoryTheory.Adjunction L R
      inst✝ : ∀ (X : D), CategoryTheory.IsSplitMono (h.counit.app X)
      X Y : D
      f : Quiver.Hom (R.obj X) (R.obj Y)
      ⊢ Exists fun a => Eq (R.map a) f
    -/
    use (retraction (h.counit.app X) ≫ (h.homEquiv (R.obj X) Y).symm f)
    /-
      case h
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      L : CategoryTheory.Functor C D
      R : CategoryTheory.Functor D C
      h : CategoryTheory.Adjunction L R
      inst✝ : ∀ (X : D), CategoryTheory.IsSplitMono (h.counit.app X)
      X Y : D
      f : Quiver.Hom (R.obj X) (R.obj Y)
      ⊢ Eq (R.map (CategoryTheory.CategoryStruct.comp (CategoryTheory.retraction (h. …
    -/
    suffices R.map (retraction (h.counit.app X)) = h.unit.app (R.obj X) by simp [this]
    /-
      case h
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      L : CategoryTheory.Functor C D
      R : CategoryTheory.Functor D C
      h : CategoryTheory.Adjunction L R
      inst✝ : ∀ (X : D), CategoryTheory.IsSplitMono (h.counit.app X)
      X Y : D
      f : Quiver.Hom (R.obj X) (R.obj Y)
      ⊢ Eq (R.map (CategoryTheory.retraction (h.counit.app X))) (h.unit.app (R.obj X))
    -/
    rw [← id_comp (R.map (retraction (h.counit.app X)))]
    simp only [Functor.id_obj, Functor.comp_obj, id_comp, ← h.right_triangle_components X,
      assoc, ← Functor.map_comp, IsSplitMono.id, Functor.map_id, comp_id]


/-- If the counit is an isomorphism, then the right adjoint is fully faithful. -/
noncomputable def fullyFaithfulROfIsIsoCounit [IsIso h.counit] : R.FullyFaithful where
  preimage {X Y} f := inv (h.counit.app X) ≫ (h.homEquiv (R.obj X) Y).symm f


instance whiskerLeft_counit_iso_of_L_fully_faithful [L.Full] [L.Faithful] :
    IsIso (whiskerLeft L h.counit) := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    L : CategoryTheory.Functor C D
    R : CategoryTheory.Functor D C
    h : CategoryTheory.Adjunction L R
    inst✝¹ : L.Full
    inst✝ : L.Faithful
    ⊢ CategoryTheory.IsIso (CategoryTheory.whiskerLeft L h.counit)
  -/
  have := h.left_triangle
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    L : CategoryTheory.Functor C D
    R : CategoryTheory.Functor D C
    h : CategoryTheory.Adjunction L R
    inst✝¹ : L.Full
    inst✝ : L.Faithful
    this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.whiskerRight h.u …
    ⊢ CategoryTheory.IsIso (CategoryTheory.whiskerLeft L h.counit)
  -/
  rw [← IsIso.eq_inv_comp] at this
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    L : CategoryTheory.Functor C D
    R : CategoryTheory.Functor D C
    h : CategoryTheory.Adjunction L R
    inst✝¹ : L.Full
    inst✝ : L.Faithful
    this : Eq (CategoryTheory.whiskerLeft L h.counit) (CategoryTheory.CategoryStru …
    ⊢ CategoryTheory.IsIso (CategoryTheory.whiskerLeft L h.counit)
  -/
  rw [this]
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    L : CategoryTheory.Functor C D
    R : CategoryTheory.Functor D C
    h : CategoryTheory.Adjunction L R
    inst✝¹ : L.Full
    inst✝ : L.Faithful
    this : Eq (CategoryTheory.whiskerLeft L h.counit) (CategoryTheory.CategoryStru …
    ⊢ CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp (CategoryTheory.inv …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance whiskerRight_counit_iso_of_L_fully_faithful [L.Full] [L.Faithful] :
    IsIso (whiskerRight h.counit R) := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    L : CategoryTheory.Functor C D
    R : CategoryTheory.Functor D C
    h : CategoryTheory.Adjunction L R
    inst✝¹ : L.Full
    inst✝ : L.Faithful
    ⊢ CategoryTheory.IsIso (CategoryTheory.whiskerRight h.counit R)
  -/
  have := h.right_triangle
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    L : CategoryTheory.Functor C D
    R : CategoryTheory.Functor D C
    h : CategoryTheory.Adjunction L R
    inst✝¹ : L.Full
    inst✝ : L.Faithful
    this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.whiskerLeft R h. …
    ⊢ CategoryTheory.IsIso (CategoryTheory.whiskerRight h.counit R)
  -/
  rw [← IsIso.eq_inv_comp] at this
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    L : CategoryTheory.Functor C D
    R : CategoryTheory.Functor D C
    h : CategoryTheory.Adjunction L R
    inst✝¹ : L.Full
    inst✝ : L.Faithful
    this : Eq (CategoryTheory.whiskerRight h.counit R) (CategoryTheory.CategoryStr …
    ⊢ CategoryTheory.IsIso (CategoryTheory.whiskerRight h.counit R)
  -/
  rw [this]
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    L : CategoryTheory.Functor C D
    R : CategoryTheory.Functor D C
    h : CategoryTheory.Adjunction L R
    inst✝¹ : L.Full
    inst✝ : L.Faithful
    this : Eq (CategoryTheory.whiskerRight h.counit R) (CategoryTheory.CategoryStr …
    ⊢ CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp (CategoryTheory.inv …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance whiskerLeft_unit_iso_of_R_fully_faithful [R.Full] [R.Faithful] :
    IsIso (whiskerLeft R h.unit) := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    L : CategoryTheory.Functor C D
    R : CategoryTheory.Functor D C
    h : CategoryTheory.Adjunction L R
    inst✝¹ : R.Full
    inst✝ : R.Faithful
    ⊢ CategoryTheory.IsIso (CategoryTheory.whiskerLeft R h.unit)
  -/
  have := h.right_triangle
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    L : CategoryTheory.Functor C D
    R : CategoryTheory.Functor D C
    h : CategoryTheory.Adjunction L R
    inst✝¹ : R.Full
    inst✝ : R.Faithful
    this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.whiskerLeft R h. …
    ⊢ CategoryTheory.IsIso (CategoryTheory.whiskerLeft R h.unit)
  -/
  rw [← IsIso.eq_comp_inv] at this
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    L : CategoryTheory.Functor C D
    R : CategoryTheory.Functor D C
    h : CategoryTheory.Adjunction L R
    inst✝¹ : R.Full
    inst✝ : R.Faithful
    this : Eq (CategoryTheory.whiskerLeft R h.unit) (CategoryTheory.CategoryStruct …
    ⊢ CategoryTheory.IsIso (CategoryTheory.whiskerLeft R h.unit)
  -/
  rw [this]
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    L : CategoryTheory.Functor C D
    R : CategoryTheory.Functor D C
    h : CategoryTheory.Adjunction L R
    inst✝¹ : R.Full
    inst✝ : R.Faithful
    this : Eq (CategoryTheory.whiskerLeft R h.unit) (CategoryTheory.CategoryStruct …
    ⊢ CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp (CategoryTheory.Cat …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance whiskerRight_unit_iso_of_R_fully_faithful [R.Full] [R.Faithful] :
    IsIso (whiskerRight h.unit L) := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    L : CategoryTheory.Functor C D
    R : CategoryTheory.Functor D C
    h : CategoryTheory.Adjunction L R
    inst✝¹ : R.Full
    inst✝ : R.Faithful
    ⊢ CategoryTheory.IsIso (CategoryTheory.whiskerRight h.unit L)
  -/
  have := h.left_triangle
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    L : CategoryTheory.Functor C D
    R : CategoryTheory.Functor D C
    h : CategoryTheory.Adjunction L R
    inst✝¹ : R.Full
    inst✝ : R.Faithful
    this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.whiskerRight h.u …
    ⊢ CategoryTheory.IsIso (CategoryTheory.whiskerRight h.unit L)
  -/
  rw [← IsIso.eq_comp_inv] at this
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    L : CategoryTheory.Functor C D
    R : CategoryTheory.Functor D C
    h : CategoryTheory.Adjunction L R
    inst✝¹ : R.Full
    inst✝ : R.Faithful
    this : Eq (CategoryTheory.whiskerRight h.unit L) (CategoryTheory.CategoryStruc …
    ⊢ CategoryTheory.IsIso (CategoryTheory.whiskerRight h.unit L)
  -/
  rw [this]
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    L : CategoryTheory.Functor C D
    R : CategoryTheory.Functor D C
    h : CategoryTheory.Adjunction L R
    inst✝¹ : R.Full
    inst✝ : R.Faithful
    this : Eq (CategoryTheory.whiskerRight h.unit L) (CategoryTheory.CategoryStruc …
    ⊢ CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp (CategoryTheory.Cat …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance [L.Faithful] [L.Full] {Y : C} : IsIso (h.counit.app (L.obj Y)) :=
  isIso_of_hom_comp_eq_id _ (h.left_triangle_components Y)


instance [L.Faithful] [L.Full] {Y : D} : IsIso (R.map (h.counit.app Y)) :=
  isIso_of_hom_comp_eq_id _ (h.right_triangle_components Y)


lemma isIso_counit_app_iff_mem_essImage [L.Faithful] [L.Full] {X : D} :
    IsIso (h.counit.app X) ↔ X ∈ L.essImage := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    L : CategoryTheory.Functor C D
    R : CategoryTheory.Functor D C
    h : CategoryTheory.Adjunction L R
    inst✝¹ : L.Faithful
    inst✝ : L.Full
    X : D
    ⊢ Iff (CategoryTheory.IsIso (h.counit.app X)) (Membership.mem L.essImage X)
  -/
  constructor
    /-
      case mp
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      L : CategoryTheory.Functor C D
      R : CategoryTheory.Functor D C
      h : CategoryTheory.Adjunction L R
      inst✝¹ : L.Faithful
      inst✝ : L.Full
      X : D
      ⊢ CategoryTheory.IsIso (h.counit.app X) → Membership.mem L.essImage X
    -/
  · intro
    /-
      case mp
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      L : CategoryTheory.Functor C D
      R : CategoryTheory.Functor D C
      h : CategoryTheory.Adjunction L R
      inst✝¹ : L.Faithful
      inst✝ : L.Full
      X : D
      a✝ : CategoryTheory.IsIso (h.counit.app X)
      ⊢ Membership.mem L.essImage X
    -/
    exact ⟨R.obj X, ⟨asIso (h.counit.app X)⟩⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      L : CategoryTheory.Functor C D
      R : CategoryTheory.Functor D C
      h : CategoryTheory.Adjunction L R
      inst✝¹ : L.Faithful
      inst✝ : L.Full
      X : D
      ⊢ Membership.mem L.essImage X → CategoryTheory.IsIso (h.counit.app X)
    -/
  · rintro ⟨_, ⟨i⟩⟩
    /-
      case mpr.intro.intro
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      L : CategoryTheory.Functor C D
      R : CategoryTheory.Functor D C
      h : CategoryTheory.Adjunction L R
      inst✝¹ : L.Faithful
      inst✝ : L.Full
      X : D
      w✝ : C
      i : CategoryTheory.Iso (L.obj w✝) X
      ⊢ CategoryTheory.IsIso (h.counit.app X)
    -/
    rw [NatTrans.isIso_app_iff_of_iso _ i.symm]
    /-
      case mpr.intro.intro
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      L : CategoryTheory.Functor C D
      R : CategoryTheory.Functor D C
      h : CategoryTheory.Adjunction L R
      inst✝¹ : L.Faithful
      inst✝ : L.Full
      X : D
      w✝ : C
      i : CategoryTheory.Iso (L.obj w✝) X
      ⊢ CategoryTheory.IsIso (h.counit.app (L.obj w✝))
    -/
    infer_instance
    /-
      🎉 no goals
    -/


lemma mem_essImage_of_counit_isIso (A : D)
    [IsIso (h.counit.app A)] : A ∈ L.essImage :=
  ⟨R.obj A, ⟨asIso (h.counit.app A)⟩⟩


lemma isIso_counit_app_of_iso [L.Faithful] [L.Full] {X : D} {Y : C} (e : X ≅ L.obj Y) :
    IsIso (h.counit.app X) :=
  (isIso_counit_app_iff_mem_essImage h).mpr ⟨Y, ⟨e.symm⟩⟩


instance [R.Faithful] [R.Full] {Y : D} : IsIso (h.unit.app (R.obj Y)) :=
  isIso_of_comp_hom_eq_id _ (h.right_triangle_components Y)


instance [R.Faithful] [R.Full] {X : C} : IsIso (L.map (h.unit.app X)) :=
  isIso_of_comp_hom_eq_id _ (h.left_triangle_components X)


lemma isIso_unit_app_iff_mem_essImage [R.Faithful] [R.Full] {Y : C} :
    IsIso (h.unit.app Y) ↔ Y ∈ R.essImage := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    L : CategoryTheory.Functor C D
    R : CategoryTheory.Functor D C
    h : CategoryTheory.Adjunction L R
    inst✝¹ : R.Faithful
    inst✝ : R.Full
    Y : C
    ⊢ Iff (CategoryTheory.IsIso (h.unit.app Y)) (Membership.mem R.essImage Y)
  -/
  constructor
    /-
      case mp
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      L : CategoryTheory.Functor C D
      R : CategoryTheory.Functor D C
      h : CategoryTheory.Adjunction L R
      inst✝¹ : R.Faithful
      inst✝ : R.Full
      Y : C
      ⊢ CategoryTheory.IsIso (h.unit.app Y) → Membership.mem R.essImage Y
    -/
  · intro
    /-
      case mp
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      L : CategoryTheory.Functor C D
      R : CategoryTheory.Functor D C
      h : CategoryTheory.Adjunction L R
      inst✝¹ : R.Faithful
      inst✝ : R.Full
      Y : C
      a✝ : CategoryTheory.IsIso (h.unit.app Y)
      ⊢ Membership.mem R.essImage Y
    -/
    exact ⟨L.obj Y, ⟨(asIso (h.unit.app Y)).symm⟩⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      L : CategoryTheory.Functor C D
      R : CategoryTheory.Functor D C
      h : CategoryTheory.Adjunction L R
      inst✝¹ : R.Faithful
      inst✝ : R.Full
      Y : C
      ⊢ Membership.mem R.essImage Y → CategoryTheory.IsIso (h.unit.app Y)
    -/
  · rintro ⟨_, ⟨i⟩⟩
    /-
      case mpr.intro.intro
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      L : CategoryTheory.Functor C D
      R : CategoryTheory.Functor D C
      h : CategoryTheory.Adjunction L R
      inst✝¹ : R.Faithful
      inst✝ : R.Full
      Y : C
      w✝ : D
      i : CategoryTheory.Iso (R.obj w✝) Y
      ⊢ CategoryTheory.IsIso (h.unit.app Y)
    -/
    rw [NatTrans.isIso_app_iff_of_iso _ i.symm]
    /-
      case mpr.intro.intro
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      L : CategoryTheory.Functor C D
      R : CategoryTheory.Functor D C
      h : CategoryTheory.Adjunction L R
      inst✝¹ : R.Faithful
      inst✝ : R.Full
      Y : C
      w✝ : D
      i : CategoryTheory.Iso (R.obj w✝) Y
      ⊢ CategoryTheory.IsIso (h.unit.app (R.obj w✝))
    -/
    infer_instance
    /-
      🎉 no goals
    -/


/-- If `η_A` is an isomorphism, then `A` is in the essential image of `i`. -/
theorem mem_essImage_of_unit_isIso (A : C)
    [IsIso (h.unit.app A)] : A ∈ R.essImage :=
  ⟨L.obj A, ⟨(asIso (h.unit.app A)).symm⟩⟩


@[deprecated (since := "2024-06-19")] alias _root_.CategoryTheory.mem_essImage_of_unit_isIso :=
  mem_essImage_of_unit_isIso


lemma isIso_unit_app_of_iso [R.Faithful] [R.Full] {X : D} {Y : C} (e : Y ≅ R.obj X) :
    IsIso (h.unit.app Y) :=
  (isIso_unit_app_iff_mem_essImage h).mpr ⟨X, ⟨e.symm⟩⟩


instance [R.IsEquivalence] : IsIso h.unit := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    L : CategoryTheory.Functor C D
    R : CategoryTheory.Functor D C
    h : CategoryTheory.Adjunction L R
    inst✝ : R.IsEquivalence
    ⊢ CategoryTheory.IsIso h.unit
  -/
  have := fun Y => isIso_unit_app_of_iso h (R.objObjPreimageIso Y).symm
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    L : CategoryTheory.Functor C D
    R : CategoryTheory.Functor D C
    h : CategoryTheory.Adjunction L R
    inst✝ : R.IsEquivalence
    this : ∀ (Y : C), CategoryTheory.IsIso (h.unit.app Y)
    ⊢ CategoryTheory.IsIso h.unit
  -/
  apply NatIso.isIso_of_isIso_app
  /-
    🎉 no goals
  -/


instance [L.IsEquivalence] : IsIso h.counit := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    L : CategoryTheory.Functor C D
    R : CategoryTheory.Functor D C
    h : CategoryTheory.Adjunction L R
    inst✝ : L.IsEquivalence
    ⊢ CategoryTheory.IsIso h.counit
  -/
  have := fun X => isIso_counit_app_of_iso h (L.objObjPreimageIso X).symm
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    L : CategoryTheory.Functor C D
    R : CategoryTheory.Functor D C
    h : CategoryTheory.Adjunction L R
    inst✝ : L.IsEquivalence
    this : ∀ (X : D), CategoryTheory.IsIso (h.counit.app X)
    ⊢ CategoryTheory.IsIso h.counit
  -/
  apply NatIso.isIso_of_isIso_app
  /-
    🎉 no goals
  -/


lemma isEquivalence_left_of_isEquivalence_right (h : L ⊣ R) [R.IsEquivalence] : L.IsEquivalence :=
  h.toEquivalence.isEquivalence_functor


lemma isEquivalence_right_of_isEquivalence_left (h : L ⊣ R) [L.IsEquivalence] : R.IsEquivalence :=
  h.toEquivalence.isEquivalence_inverse


instance [L.IsEquivalence] : IsIso h.unit := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    L : CategoryTheory.Functor C D
    R : CategoryTheory.Functor D C
    h : CategoryTheory.Adjunction L R
    inst✝ : L.IsEquivalence
    ⊢ CategoryTheory.IsIso h.unit
  -/
  have := h.isEquivalence_right_of_isEquivalence_left
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    L : CategoryTheory.Functor C D
    R : CategoryTheory.Functor D C
    h : CategoryTheory.Adjunction L R
    inst✝ : L.IsEquivalence
    this : R.IsEquivalence
    ⊢ CategoryTheory.IsIso h.unit
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance [R.IsEquivalence] : IsIso h.counit := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    L : CategoryTheory.Functor C D
    R : CategoryTheory.Functor D C
    h : CategoryTheory.Adjunction L R
    inst✝ : R.IsEquivalence
    ⊢ CategoryTheory.IsIso h.counit
  -/
  have := h.isEquivalence_left_of_isEquivalence_right
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    L : CategoryTheory.Functor C D
    R : CategoryTheory.Functor D C
    h : CategoryTheory.Adjunction L R
    inst✝ : R.IsEquivalence
    this : L.IsEquivalence
    ⊢ CategoryTheory.IsIso h.counit
  -/
  infer_instance
  /-
    🎉 no goals
  -/


