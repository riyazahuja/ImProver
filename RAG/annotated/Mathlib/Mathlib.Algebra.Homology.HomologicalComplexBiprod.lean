instance (i : ι) : HasBinaryBiproduct ((eval C c i).obj K) ((eval C c i).obj L) := by
  /-
    C : Type u_1
    ι : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    c : ComplexShape ι
    K L : HomologicalComplex C c
    inst✝ : ∀ (i : ι), CategoryTheory.Limits.HasBinaryBiproduct (K.X i) (L.X i)
    i : ι
    ⊢ CategoryTheory.Limits.HasBinaryBiproduct ((HomologicalComplex.eval C c i).ob …
  -/
  dsimp [eval]
  /-
    C : Type u_1
    ι : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    c : ComplexShape ι
    K L : HomologicalComplex C c
    inst✝ : ∀ (i : ι), CategoryTheory.Limits.HasBinaryBiproduct (K.X i) (L.X i)
    i : ι
    ⊢ CategoryTheory.Limits.HasBinaryBiproduct (K.X i) (L.X i)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance (i : ι) : HasLimit ((pair K L) ⋙ (eval C c i)) := by
  /-
    C : Type u_1
    ι : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    c : ComplexShape ι
    K L : HomologicalComplex C c
    inst✝ : ∀ (i : ι), CategoryTheory.Limits.HasBinaryBiproduct (K.X i) (L.X i)
    i : ι
    ⊢ CategoryTheory.Limits.HasLimit ((CategoryTheory.Limits.pair K L).comp (Homol …
  -/
  have e : _ ≅ pair (K.X i) (L.X i) := diagramIsoPair (pair K L ⋙ eval C c i)
  /-
    C : Type u_1
    ι : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    c : ComplexShape ι
    K L : HomologicalComplex C c
    inst✝ : ∀ (i : ι), CategoryTheory.Limits.HasBinaryBiproduct (K.X i) (L.X i)
    i : ι
    e : CategoryTheory.Iso ((CategoryTheory.Limits.pair K L).comp (HomologicalComp …
    ⊢ CategoryTheory.Limits.HasLimit ((CategoryTheory.Limits.pair K L).comp (Homol …
  -/
  exact hasLimitOfIso e.symm
  /-
    🎉 no goals
  -/


instance (i : ι) : HasColimit ((pair K L) ⋙ (eval C c i)) := by
  /-
    C : Type u_1
    ι : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    c : ComplexShape ι
    K L : HomologicalComplex C c
    inst✝ : ∀ (i : ι), CategoryTheory.Limits.HasBinaryBiproduct (K.X i) (L.X i)
    i : ι
    ⊢ CategoryTheory.Limits.HasColimit ((CategoryTheory.Limits.pair K L).comp (Hom …
  -/
  have e : _ ≅ pair (K.X i) (L.X i) := diagramIsoPair (pair K L ⋙ eval C c i)
  /-
    C : Type u_1
    ι : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    c : ComplexShape ι
    K L : HomologicalComplex C c
    inst✝ : ∀ (i : ι), CategoryTheory.Limits.HasBinaryBiproduct (K.X i) (L.X i)
    i : ι
    e : CategoryTheory.Iso ((CategoryTheory.Limits.pair K L).comp (HomologicalComp …
    ⊢ CategoryTheory.Limits.HasColimit ((CategoryTheory.Limits.pair K L).comp (Hom …
  -/
  exact hasColimitOfIso e
  /-
    🎉 no goals
  -/


instance : HasBinaryBiproduct K L := HasBinaryBiproduct.of_hasBinaryProduct _ _


instance (i : ι) : PreservesBinaryBiproduct K L (eval C c i) :=
  preservesBinaryBiproduct_of_preservesBinaryProduct _


/-- The canonical isomorphism `(K ⊞ L).X i ≅ (K.X i) ⊞ (L.X i)`. -/
noncomputable def biprodXIso (i : ι) : (K ⊞ L).X i ≅ (K.X i) ⊞ (L.X i) :=
  (eval C c i).mapBiprod K L


@[reassoc (attr := simp)]
lemma inl_biprodXIso_inv (i : ι) :
    biprod.inl ≫ (biprodXIso K L i).inv = (biprod.inl : K ⟶ K ⊞ L).f i := by
  /-
    C : Type u_1
    ι : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    c : ComplexShape ι
    K L : HomologicalComplex C c
    inst✝ : ∀ (i : ι), CategoryTheory.Limits.HasBinaryBiproduct (K.X i) (L.X i)
    i : ι
    ⊢ Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.Limits.biprod.inl (K.b …
  -/
  simp [biprodXIso]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma inr_biprodXIso_inv (i : ι) :
    biprod.inr ≫ (biprodXIso K L i).inv = (biprod.inr : L ⟶ K ⊞ L).f i := by
  /-
    C : Type u_1
    ι : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    c : ComplexShape ι
    K L : HomologicalComplex C c
    inst✝ : ∀ (i : ι), CategoryTheory.Limits.HasBinaryBiproduct (K.X i) (L.X i)
    i : ι
    ⊢ Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.Limits.biprod.inr (K.b …
  -/
  simp [biprodXIso]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma biprodXIso_hom_fst (i : ι) :
    (biprodXIso K L i).hom ≫ biprod.fst = (biprod.fst : K ⊞ L ⟶ K).f i := by
  /-
    C : Type u_1
    ι : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    c : ComplexShape ι
    K L : HomologicalComplex C c
    inst✝ : ∀ (i : ι), CategoryTheory.Limits.HasBinaryBiproduct (K.X i) (L.X i)
    i : ι
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.biprodXIso L i).hom CategoryTheory …
  -/
  simp [biprodXIso]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma biprodXIso_hom_snd (i : ι) :
    (biprodXIso K L i).hom ≫ biprod.snd = (biprod.snd : K ⊞ L ⟶ L).f i := by
  /-
    C : Type u_1
    ι : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    c : ComplexShape ι
    K L : HomologicalComplex C c
    inst✝ : ∀ (i : ι), CategoryTheory.Limits.HasBinaryBiproduct (K.X i) (L.X i)
    i : ι
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.biprodXIso L i).hom CategoryTheory …
  -/
  simp [biprodXIso]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma biprod_inl_fst_f (i : ι) :
    (biprod.inl : K ⟶ K ⊞ L).f i ≫ (biprod.fst : K ⊞ L ⟶ K).f i = 𝟙 _ := by
  /-
    C : Type u_1
    ι : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    c : ComplexShape ι
    K L : HomologicalComplex C c
    inst✝ : ∀ (i : ι), CategoryTheory.Limits.HasBinaryBiproduct (K.X i) (L.X i)
    i : ι
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.biprod.inl.f i …
  -/
  rw [← comp_f, biprod.inl_fst, id_f]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma biprod_inl_snd_f (i : ι) :
    (biprod.inl : K ⟶ K ⊞ L).f i ≫ (biprod.snd : K ⊞ L ⟶ L).f i = 0 := by
  /-
    C : Type u_1
    ι : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    c : ComplexShape ι
    K L : HomologicalComplex C c
    inst✝ : ∀ (i : ι), CategoryTheory.Limits.HasBinaryBiproduct (K.X i) (L.X i)
    i : ι
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.biprod.inl.f i …
  -/
  rw [← comp_f, biprod.inl_snd, zero_f]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma biprod_inr_fst_f (i : ι) :
    (biprod.inr : L ⟶ K ⊞ L).f i ≫ (biprod.fst : K ⊞ L ⟶ K).f i = 0 := by
  /-
    C : Type u_1
    ι : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    c : ComplexShape ι
    K L : HomologicalComplex C c
    inst✝ : ∀ (i : ι), CategoryTheory.Limits.HasBinaryBiproduct (K.X i) (L.X i)
    i : ι
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.biprod.inr.f i …
  -/
  rw [← comp_f, biprod.inr_fst, zero_f]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma biprod_inr_snd_f (i : ι) :
    (biprod.inr : L ⟶ K ⊞ L).f i ≫ (biprod.snd : K ⊞ L ⟶ L).f i = 𝟙 _ := by
  /-
    C : Type u_1
    ι : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    c : ComplexShape ι
    K L : HomologicalComplex C c
    inst✝ : ∀ (i : ι), CategoryTheory.Limits.HasBinaryBiproduct (K.X i) (L.X i)
    i : ι
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.biprod.inr.f i …
  -/
  rw [← comp_f, biprod.inr_snd, id_f]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma biprod_inl_desc_f (α : K ⟶ M) (β : L ⟶ M) (i : ι) :
    (biprod.inl : K ⟶ K ⊞ L).f i ≫ (biprod.desc α β).f i = α.f i := by
  /-
    C : Type u_1
    ι : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    c : ComplexShape ι
    K L : HomologicalComplex C c
    inst✝ : ∀ (i : ι), CategoryTheory.Limits.HasBinaryBiproduct (K.X i) (L.X i)
    M : HomologicalComplex C c
    α : Quiver.Hom K M
    β : Quiver.Hom L M
    i : ι
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.biprod.inl.f i …
  -/
  rw [← comp_f, biprod.inl_desc]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma biprod_inr_desc_f (α : K ⟶ M) (β : L ⟶ M) (i : ι) :
    (biprod.inr : L ⟶ K ⊞ L).f i ≫ (biprod.desc α β).f i = β.f i := by
  /-
    C : Type u_1
    ι : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    c : ComplexShape ι
    K L : HomologicalComplex C c
    inst✝ : ∀ (i : ι), CategoryTheory.Limits.HasBinaryBiproduct (K.X i) (L.X i)
    M : HomologicalComplex C c
    α : Quiver.Hom K M
    β : Quiver.Hom L M
    i : ι
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.biprod.inr.f i …
  -/
  rw [← comp_f, biprod.inr_desc]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma biprod_lift_fst_f (α : M ⟶ K) (β : M ⟶ L) (i : ι) :
    (biprod.lift α β).f i ≫ (biprod.fst : K ⊞ L ⟶ K).f i = α.f i := by
  /-
    C : Type u_1
    ι : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    c : ComplexShape ι
    K L : HomologicalComplex C c
    inst✝ : ∀ (i : ι), CategoryTheory.Limits.HasBinaryBiproduct (K.X i) (L.X i)
    M : HomologicalComplex C c
    α : Quiver.Hom M K
    β : Quiver.Hom M L
    i : ι
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.biprod.lift α …
  -/
  rw [← comp_f, biprod.lift_fst]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma biprod_lift_snd_f (α : M ⟶ K) (β : M ⟶ L) (i : ι) :
    (biprod.lift α β).f i ≫ (biprod.snd : K ⊞ L ⟶ L).f i = β.f i := by
  /-
    C : Type u_1
    ι : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    c : ComplexShape ι
    K L : HomologicalComplex C c
    inst✝ : ∀ (i : ι), CategoryTheory.Limits.HasBinaryBiproduct (K.X i) (L.X i)
    M : HomologicalComplex C c
    α : Quiver.Hom M K
    β : Quiver.Hom M L
    i : ι
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.biprod.lift α …
  -/
  rw [← comp_f, biprod.lift_snd]
  /-
    🎉 no goals
  -/


