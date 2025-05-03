/-- A morphism of homological complexes `φ : F ⟶ G` has a homotopy cofiber if for all
indices `i` and `j` such that `c.Rel i j`, the binary biproduct `F.X j ⊞ G.X i` exists. -/
class HasHomotopyCofiber (φ : F ⟶ G) : Prop where
  hasBinaryBiproduct (i j : ι) (hij : c.Rel i j) : HasBinaryBiproduct (F.X j) (G.X i)


instance [HasBinaryBiproducts C] : HasHomotopyCofiber φ where
  hasBinaryBiproduct _ _ _ := inferInstance


/-- The `X` field of the homological complex `homotopyCofiber φ`. -/
noncomputable def X (i : ι) : C :=
  if hi : c.Rel i (c.next i)
  then
    haveI := HasHomotopyCofiber.hasBinaryBiproduct φ _ _ hi
    (F.X (c.next i)) ⊞ (G.X i)
  else G.X i


/-- The canonical isomorphism `(homotopyCofiber φ).X i ≅ F.X j ⊞ G.X i` when `c.Rel i j`. -/
noncomputable def XIsoBiprod (i j : ι) (hij : c.Rel i j) [HasBinaryBiproduct (F.X j) (G.X i)] :
    X φ i ≅ F.X j ⊞ G.X i :=
  eqToIso (by
    /-
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{?u.2895, u_1} C
      inst✝³ : CategoryTheory.Preadditive C
      ι : Type u_2
      c : ComplexShape ι
      F G K : HomologicalComplex C c
      φ : Quiver.Hom F G
      inst✝² : HomologicalComplex.HasHomotopyCofiber φ
      inst✝¹ : DecidableRel c.Rel
      i j : ι
      hij : c.Rel i j
      inst✝ : CategoryTheory.Limits.HasBinaryBiproduct (F.X j) (G.X i)
      ⊢ Eq (HomologicalComplex.homotopyCofiber.X φ i) (CategoryTheory.Limits.biprod  …
    -/
    obtain rfl := c.next_eq' hij
    /-
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{?u.2895, u_1} C
      inst✝³ : CategoryTheory.Preadditive C
      ι : Type u_2
      c : ComplexShape ι
      F G K : HomologicalComplex C c
      φ : Quiver.Hom F G
      inst✝² : HomologicalComplex.HasHomotopyCofiber φ
      inst✝¹ : DecidableRel c.Rel
      i : ι
      hij : c.Rel i (c.next i)
      inst✝ : CategoryTheory.Limits.HasBinaryBiproduct (F.X (c.next i)) (G.X i)
      ⊢ Eq (HomologicalComplex.homotopyCofiber.X φ i) (CategoryTheory.Limits.biprod  …
    -/
    apply dif_pos hij)
    /-
      🎉 no goals
    -/


/-- The canonical isomorphism `(homotopyCofiber φ).X i ≅ G.X i` when `¬ c.Rel i (c.next i)`. -/
noncomputable def XIso (i : ι) (hi : ¬ c.Rel i (c.next i)) :
    X φ i ≅ G.X i :=
  eqToIso (dif_neg hi)


/-- The second projection `(homotopyCofiber φ).X i ⟶ G.X i`. -/
noncomputable def sndX (i : ι) : X φ i ⟶ G.X i :=
  if hi : c.Rel i (c.next i)
  then
    haveI := HasHomotopyCofiber.hasBinaryBiproduct φ _ _ hi
    (XIsoBiprod φ _ _ hi).hom ≫ biprod.snd
  else
    (XIso φ i hi).hom


/-- The right inclusion `G.X i ⟶ (homotopyCofiber φ).X i`. -/
noncomputable def inrX (i : ι) : G.X i ⟶ X φ i :=
  if hi : c.Rel i (c.next i)
  then
    haveI := HasHomotopyCofiber.hasBinaryBiproduct φ _ _ hi
    biprod.inr ≫ (XIsoBiprod φ _ _ hi).inv
  else
    (XIso φ i hi).inv


@[reassoc (attr := simp)]
lemma inrX_sndX (i : ι) : inrX φ i ≫ sndX φ i = 𝟙 _ := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    ι : Type u_2
    c : ComplexShape ι
    F G : HomologicalComplex C c
    φ : Quiver.Hom F G
    inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
    inst✝ : DecidableRel c.Rel
    i : ι
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.homotopyCofiber.i …
  -/
  dsimp [sndX, inrX]
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    ι : Type u_2
    c : ComplexShape ι
    F G : HomologicalComplex C c
    φ : Quiver.Hom F G
    inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
    inst✝ : DecidableRel c.Rel
    i : ι
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (dite (c.Rel i (c.next i)) (fun hi => …
  -/
                        /-
                          🎉 no goals
                        -/
  split_ifs with hi <;> simp
                        /-
                          🎉 no goals
                        -/


@[reassoc]
lemma sndX_inrX (i : ι) (hi : ¬ c.Rel i (c.next i)) :
    sndX φ i ≫ inrX φ i = 𝟙 _ := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    ι : Type u_2
    c : ComplexShape ι
    F G : HomologicalComplex C c
    φ : Quiver.Hom F G
    inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
    inst✝ : DecidableRel c.Rel
    i : ι
    hi : Not (c.Rel i (c.next i))
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.homotopyCofiber.s …
  -/
  dsimp [sndX, inrX]
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    ι : Type u_2
    c : ComplexShape ι
    F G : HomologicalComplex C c
    φ : Quiver.Hom F G
    inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
    inst✝ : DecidableRel c.Rel
    i : ι
    hi : Not (c.Rel i (c.next i))
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (dite (c.Rel i (c.next i)) (fun hi => …
  -/
  simp only [dif_neg hi, Iso.hom_inv_id]
  /-
    🎉 no goals
  -/


/-- The first projection `(homotopyCofiber φ).X i ⟶ F.X j` when `c.Rel i j`. -/
noncomputable def fstX (i j : ι) (hij : c.Rel i j) : X φ i ⟶ F.X j :=
  haveI := HasHomotopyCofiber.hasBinaryBiproduct φ _ _ hij
  (XIsoBiprod φ i j hij).hom ≫ biprod.fst


/-- The left inclusion `F.X i ⟶ (homotopyCofiber φ).X j` when `c.Rel j i`. -/
noncomputable def inlX (i j : ι) (hij : c.Rel j i) : F.X i ⟶ X φ j :=
  haveI := HasHomotopyCofiber.hasBinaryBiproduct φ _ _ hij
  biprod.inl ≫ (XIsoBiprod φ j i hij).inv


@[reassoc (attr := simp)]
lemma inlX_fstX (i j : ι ) (hij : c.Rel j i) :
    inlX φ i j hij ≫ fstX φ j i hij = 𝟙 _ := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    ι : Type u_2
    c : ComplexShape ι
    F G : HomologicalComplex C c
    φ : Quiver.Hom F G
    inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
    inst✝ : DecidableRel c.Rel
    i j : ι
    hij : c.Rel j i
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.homotopyCofiber.i …
  -/
  simp [inlX, fstX]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma inlX_sndX (i j : ι) (hij : c.Rel j i) :
    inlX φ i j hij ≫ sndX φ j = 0 := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    ι : Type u_2
    c : ComplexShape ι
    F G : HomologicalComplex C c
    φ : Quiver.Hom F G
    inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
    inst✝ : DecidableRel c.Rel
    i j : ι
    hij : c.Rel j i
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.homotopyCofiber.i …
  -/
  obtain rfl := c.next_eq' hij
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    ι : Type u_2
    c : ComplexShape ι
    F G : HomologicalComplex C c
    φ : Quiver.Hom F G
    inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
    inst✝ : DecidableRel c.Rel
    j : ι
    hij : c.Rel j (c.next j)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.homotopyCofiber.i …
  -/
  simp [inlX, sndX, dif_pos hij]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma inrX_fstX (i j : ι) (hij : c.Rel i j) :
    inrX φ i ≫ fstX φ i j hij = 0 := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    ι : Type u_2
    c : ComplexShape ι
    F G : HomologicalComplex C c
    φ : Quiver.Hom F G
    inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
    inst✝ : DecidableRel c.Rel
    i j : ι
    hij : c.Rel i j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.homotopyCofiber.i …
  -/
  obtain rfl := c.next_eq' hij
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    ι : Type u_2
    c : ComplexShape ι
    F G : HomologicalComplex C c
    φ : Quiver.Hom F G
    inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
    inst✝ : DecidableRel c.Rel
    i : ι
    hij : c.Rel i (c.next i)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.homotopyCofiber.i …
  -/
  simp [inrX, fstX, dif_pos hij]
  /-
    🎉 no goals
  -/


/-- The `d` field of the homological complex `homotopyCofiber φ`. -/
noncomputable def d (i j : ι) : X φ i ⟶ X φ j :=
  if hij : c.Rel i j
  then
    (if hj : c.Rel j (c.next j) then -fstX φ i j hij ≫ F.d _ _ ≫ inlX φ _ _ hj else 0) +
      fstX φ i j hij ≫ φ.f j ≫ inrX φ j + sndX φ i ≫ G.d i j ≫ inrX φ j
  else
    0


lemma ext_to_X (i j : ι) (hij : c.Rel i j) {A : C} {f g : A ⟶ X φ i}
    (h₁ : f ≫ fstX φ i j hij = g ≫ fstX φ i j hij) (h₂ : f ≫ sndX φ i = g ≫ sndX φ i) :
    f = g := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    ι : Type u_2
    c : ComplexShape ι
    F G : HomologicalComplex C c
    φ : Quiver.Hom F G
    inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
    inst✝ : DecidableRel c.Rel
    i j : ι
    hij : c.Rel i j
    A : C
    f g : Quiver.Hom A (HomologicalComplex.homotopyCofiber.X φ i)
    h₁ : Eq (CategoryTheory.CategoryStruct.comp f (HomologicalComplex.homotopyCofi …
    h₂ : Eq (CategoryTheory.CategoryStruct.comp f (HomologicalComplex.homotopyCofi …
    ⊢ Eq f g
  -/
  haveI := HasHomotopyCofiber.hasBinaryBiproduct φ _ _ hij
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    ι : Type u_2
    c : ComplexShape ι
    F G : HomologicalComplex C c
    φ : Quiver.Hom F G
    inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
    inst✝ : DecidableRel c.Rel
    i j : ι
    hij : c.Rel i j
    A : C
    f g : Quiver.Hom A (HomologicalComplex.homotopyCofiber.X φ i)
    h₁ : Eq (CategoryTheory.CategoryStruct.comp f (HomologicalComplex.homotopyCofi …
    h₂ : Eq (CategoryTheory.CategoryStruct.comp f (HomologicalComplex.homotopyCofi …
    this : CategoryTheory.Limits.HasBinaryBiproduct (F.X j) (G.X i)
    ⊢ Eq f g
  -/
  rw [← cancel_mono (XIsoBiprod φ i j hij).hom]
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    ι : Type u_2
    c : ComplexShape ι
    F G : HomologicalComplex C c
    φ : Quiver.Hom F G
    inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
    inst✝ : DecidableRel c.Rel
    i j : ι
    hij : c.Rel i j
    A : C
    f g : Quiver.Hom A (HomologicalComplex.homotopyCofiber.X φ i)
    h₁ : Eq (CategoryTheory.CategoryStruct.comp f (HomologicalComplex.homotopyCofi …
    h₂ : Eq (CategoryTheory.CategoryStruct.comp f (HomologicalComplex.homotopyCofi …
    this : CategoryTheory.Limits.HasBinaryBiproduct (F.X j) (G.X i)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp f (HomologicalComplex.homotopyCofiber …
  -/
  apply biprod.hom_ext
    /-
      case h₀
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      inst✝² : CategoryTheory.Preadditive C
      ι : Type u_2
      c : ComplexShape ι
      F G : HomologicalComplex C c
      φ : Quiver.Hom F G
      inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
      inst✝ : DecidableRel c.Rel
      i j : ι
      hij : c.Rel i j
      A : C
      f g : Quiver.Hom A (HomologicalComplex.homotopyCofiber.X φ i)
      h₁ : Eq (CategoryTheory.CategoryStruct.comp f (HomologicalComplex.homotopyCofi …
      h₂ : Eq (CategoryTheory.CategoryStruct.comp f (HomologicalComplex.homotopyCofi …
      this : CategoryTheory.Limits.HasBinaryBiproduct (F.X j) (G.X i)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp f …
    -/
  · simpa using h₁
    /-
      🎉 no goals
    -/
    /-
      case h₁
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      inst✝² : CategoryTheory.Preadditive C
      ι : Type u_2
      c : ComplexShape ι
      F G : HomologicalComplex C c
      φ : Quiver.Hom F G
      inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
      inst✝ : DecidableRel c.Rel
      i j : ι
      hij : c.Rel i j
      A : C
      f g : Quiver.Hom A (HomologicalComplex.homotopyCofiber.X φ i)
      h₁ : Eq (CategoryTheory.CategoryStruct.comp f (HomologicalComplex.homotopyCofi …
      h₂ : Eq (CategoryTheory.CategoryStruct.comp f (HomologicalComplex.homotopyCofi …
      this : CategoryTheory.Limits.HasBinaryBiproduct (F.X j) (G.X i)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp f …
    -/
  · obtain rfl := c.next_eq' hij
    /-
      case h₁
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      inst✝² : CategoryTheory.Preadditive C
      ι : Type u_2
      c : ComplexShape ι
      F G : HomologicalComplex C c
      φ : Quiver.Hom F G
      inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
      inst✝ : DecidableRel c.Rel
      i : ι
      A : C
      f g : Quiver.Hom A (HomologicalComplex.homotopyCofiber.X φ i)
      h₂ : Eq (CategoryTheory.CategoryStruct.comp f (HomologicalComplex.homotopyCofi …
      hij : c.Rel i (c.next i)
      h₁ : Eq (CategoryTheory.CategoryStruct.comp f (HomologicalComplex.homotopyCofi …
      this : CategoryTheory.Limits.HasBinaryBiproduct (F.X (c.next i)) (G.X i)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp f …
    -/
    simpa [sndX, dif_pos hij] using h₂
    /-
      🎉 no goals
    -/


lemma ext_to_X' (i : ι) (hi : ¬ c.Rel i (c.next i)) {A : C} {f g : A ⟶ X φ i}
    (h : f ≫ sndX φ i = g ≫ sndX φ i) : f = g := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    ι : Type u_2
    c : ComplexShape ι
    F G : HomologicalComplex C c
    φ : Quiver.Hom F G
    inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
    inst✝ : DecidableRel c.Rel
    i : ι
    hi : Not (c.Rel i (c.next i))
    A : C
    f g : Quiver.Hom A (HomologicalComplex.homotopyCofiber.X φ i)
    h : Eq (CategoryTheory.CategoryStruct.comp f (HomologicalComplex.homotopyCofib …
    ⊢ Eq f g
  -/
  rw [← cancel_mono (XIso φ i hi).hom]
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    ι : Type u_2
    c : ComplexShape ι
    F G : HomologicalComplex C c
    φ : Quiver.Hom F G
    inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
    inst✝ : DecidableRel c.Rel
    i : ι
    hi : Not (c.Rel i (c.next i))
    A : C
    f g : Quiver.Hom A (HomologicalComplex.homotopyCofiber.X φ i)
    h : Eq (CategoryTheory.CategoryStruct.comp f (HomologicalComplex.homotopyCofib …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp f (HomologicalComplex.homotopyCofiber …
  -/
  simpa only [sndX, dif_neg hi] using h
  /-
    🎉 no goals
  -/


lemma ext_from_X (i j : ι) (hij : c.Rel j i) {A : C} {f g : X φ j ⟶ A}
    (h₁ : inlX φ i j hij ≫ f = inlX φ i j hij ≫ g) (h₂ : inrX φ j ≫ f = inrX φ j ≫ g) :
    f = g := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    ι : Type u_2
    c : ComplexShape ι
    F G : HomologicalComplex C c
    φ : Quiver.Hom F G
    inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
    inst✝ : DecidableRel c.Rel
    i j : ι
    hij : c.Rel j i
    A : C
    f g : Quiver.Hom (HomologicalComplex.homotopyCofiber.X φ j) A
    h₁ : Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.homotopyCofibe …
    h₂ : Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.homotopyCofibe …
    ⊢ Eq f g
  -/
  haveI := HasHomotopyCofiber.hasBinaryBiproduct φ _ _ hij
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    ι : Type u_2
    c : ComplexShape ι
    F G : HomologicalComplex C c
    φ : Quiver.Hom F G
    inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
    inst✝ : DecidableRel c.Rel
    i j : ι
    hij : c.Rel j i
    A : C
    f g : Quiver.Hom (HomologicalComplex.homotopyCofiber.X φ j) A
    h₁ : Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.homotopyCofibe …
    h₂ : Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.homotopyCofibe …
    this : CategoryTheory.Limits.HasBinaryBiproduct (F.X i) (G.X j)
    ⊢ Eq f g
  -/
  rw [← cancel_epi (XIsoBiprod φ j i hij).inv]
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    ι : Type u_2
    c : ComplexShape ι
    F G : HomologicalComplex C c
    φ : Quiver.Hom F G
    inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
    inst✝ : DecidableRel c.Rel
    i j : ι
    hij : c.Rel j i
    A : C
    f g : Quiver.Hom (HomologicalComplex.homotopyCofiber.X φ j) A
    h₁ : Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.homotopyCofibe …
    h₂ : Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.homotopyCofibe …
    this : CategoryTheory.Limits.HasBinaryBiproduct (F.X i) (G.X j)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.homotopyCofiber.X …
  -/
  apply biprod.hom_ext'
    /-
      case h₀
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      inst✝² : CategoryTheory.Preadditive C
      ι : Type u_2
      c : ComplexShape ι
      F G : HomologicalComplex C c
      φ : Quiver.Hom F G
      inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
      inst✝ : DecidableRel c.Rel
      i j : ι
      hij : c.Rel j i
      A : C
      f g : Quiver.Hom (HomologicalComplex.homotopyCofiber.X φ j) A
      h₁ : Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.homotopyCofibe …
      h₂ : Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.homotopyCofibe …
      this : CategoryTheory.Limits.HasBinaryBiproduct (F.X i) (G.X j)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.Limits.biprod.inl (Cat …
    -/
  · simpa [inlX] using h₁
    /-
      🎉 no goals
    -/
    /-
      case h₁
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      inst✝² : CategoryTheory.Preadditive C
      ι : Type u_2
      c : ComplexShape ι
      F G : HomologicalComplex C c
      φ : Quiver.Hom F G
      inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
      inst✝ : DecidableRel c.Rel
      i j : ι
      hij : c.Rel j i
      A : C
      f g : Quiver.Hom (HomologicalComplex.homotopyCofiber.X φ j) A
      h₁ : Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.homotopyCofibe …
      h₂ : Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.homotopyCofibe …
      this : CategoryTheory.Limits.HasBinaryBiproduct (F.X i) (G.X j)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.Limits.biprod.inr (Cat …
    -/
  · obtain rfl := c.next_eq' hij
    /-
      case h₁
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      inst✝² : CategoryTheory.Preadditive C
      ι : Type u_2
      c : ComplexShape ι
      F G : HomologicalComplex C c
      φ : Quiver.Hom F G
      inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
      inst✝ : DecidableRel c.Rel
      j : ι
      A : C
      f g : Quiver.Hom (HomologicalComplex.homotopyCofiber.X φ j) A
      h₂ : Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.homotopyCofibe …
      hij : c.Rel j (c.next j)
      h₁ : Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.homotopyCofibe …
      this : CategoryTheory.Limits.HasBinaryBiproduct (F.X (c.next j)) (G.X j)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.Limits.biprod.inr (Cat …
    -/
    simpa [inrX, dif_pos hij] using h₂
    /-
      🎉 no goals
    -/


lemma ext_from_X' (i : ι) (hi : ¬ c.Rel i (c.next i)) {A : C} {f g : X φ i ⟶ A}
    (h : inrX φ i ≫ f = inrX φ i ≫ g) : f = g := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    ι : Type u_2
    c : ComplexShape ι
    F G : HomologicalComplex C c
    φ : Quiver.Hom F G
    inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
    inst✝ : DecidableRel c.Rel
    i : ι
    hi : Not (c.Rel i (c.next i))
    A : C
    f g : Quiver.Hom (HomologicalComplex.homotopyCofiber.X φ i) A
    h : Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.homotopyCofiber …
    ⊢ Eq f g
  -/
  rw [← cancel_epi (XIso φ i hi).inv]
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    ι : Type u_2
    c : ComplexShape ι
    F G : HomologicalComplex C c
    φ : Quiver.Hom F G
    inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
    inst✝ : DecidableRel c.Rel
    i : ι
    hi : Not (c.Rel i (c.next i))
    A : C
    f g : Quiver.Hom (HomologicalComplex.homotopyCofiber.X φ i) A
    h : Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.homotopyCofiber …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.homotopyCofiber.X …
  -/
  simpa only [inrX, dif_neg hi] using h
  /-
    🎉 no goals
  -/


@[reassoc]
lemma d_fstX (i j k : ι) (hij : c.Rel i j) (hjk : c.Rel j k) :
    d φ i j ≫ fstX φ j k hjk = -fstX φ i j hij ≫ F.d j k := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    ι : Type u_2
    c : ComplexShape ι
    F G : HomologicalComplex C c
    φ : Quiver.Hom F G
    inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
    inst✝ : DecidableRel c.Rel
    i j k : ι
    hij : c.Rel i j
    hjk : c.Rel j k
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.homotopyCofiber.d …
  -/
  obtain rfl := c.next_eq' hjk
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    ι : Type u_2
    c : ComplexShape ι
    F G : HomologicalComplex C c
    φ : Quiver.Hom F G
    inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
    inst✝ : DecidableRel c.Rel
    i j : ι
    hij : c.Rel i j
    hjk : c.Rel j (c.next j)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.homotopyCofiber.d …
  -/
  simp [d, dif_pos hij, dif_pos hjk]
  /-
    🎉 no goals
  -/


@[reassoc]
lemma d_sndX (i j : ι) (hij : c.Rel i j) :
    d φ i j ≫ sndX φ j = fstX φ i j hij ≫ φ.f j + sndX φ i ≫ G.d i j := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    ι : Type u_2
    c : ComplexShape ι
    F G : HomologicalComplex C c
    φ : Quiver.Hom F G
    inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
    inst✝ : DecidableRel c.Rel
    i j : ι
    hij : c.Rel i j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.homotopyCofiber.d …
  -/
  dsimp [d]
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    ι : Type u_2
    c : ComplexShape ι
    F G : HomologicalComplex C c
    φ : Quiver.Hom F G
    inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
    inst✝ : DecidableRel c.Rel
    i j : ι
    hij : c.Rel i j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (dite (c.Rel i j) (fun hij => HAdd.hA …
  -/
                         /-
                           🎉 no goals
                         -/
  split_ifs with hij <;> simp
                         /-
                           🎉 no goals
                         -/


@[reassoc]
lemma inlX_d (i j k : ι) (hij : c.Rel i j) (hjk : c.Rel j k) :
    inlX φ j i hij ≫ d φ i j = -F.d j k ≫ inlX φ k j hjk + φ.f j ≫ inrX φ j := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    ι : Type u_2
    c : ComplexShape ι
    F G : HomologicalComplex C c
    φ : Quiver.Hom F G
    inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
    inst✝ : DecidableRel c.Rel
    i j k : ι
    hij : c.Rel i j
    hjk : c.Rel j k
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.homotopyCofiber.i …
  -/
  apply ext_to_X φ j k hjk
    /-
      case h₁
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      inst✝² : CategoryTheory.Preadditive C
      ι : Type u_2
      c : ComplexShape ι
      F G : HomologicalComplex C c
      φ : Quiver.Hom F G
      inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
      inst✝ : DecidableRel c.Rel
      i j k : ι
      hij : c.Rel i j
      hjk : c.Rel j k
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
  · dsimp
    /-
      case h₁
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      inst✝² : CategoryTheory.Preadditive C
      ι : Type u_2
      c : ComplexShape ι
      F G : HomologicalComplex C c
      φ : Quiver.Hom F G
      inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
      inst✝ : DecidableRel c.Rel
      i j k : ι
      hij : c.Rel i j
      hjk : c.Rel j k
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    simp [d_fstX φ  _ _ _ hij hjk]
    /-
      🎉 no goals
    -/
    /-
      case h₂
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      inst✝² : CategoryTheory.Preadditive C
      ι : Type u_2
      c : ComplexShape ι
      F G : HomologicalComplex C c
      φ : Quiver.Hom F G
      inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
      inst✝ : DecidableRel c.Rel
      i j k : ι
      hij : c.Rel i j
      hjk : c.Rel j k
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
  · simp [d_sndX φ _ _ hij]
    /-
      🎉 no goals
    -/


@[reassoc]
lemma inlX_d' (i j : ι) (hij : c.Rel i j) (hj : ¬ c.Rel j (c.next j)) :
    inlX φ j i hij ≫ d φ i j = φ.f j ≫ inrX φ j := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    ι : Type u_2
    c : ComplexShape ι
    F G : HomologicalComplex C c
    φ : Quiver.Hom F G
    inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
    inst✝ : DecidableRel c.Rel
    i j : ι
    hij : c.Rel i j
    hj : Not (c.Rel j (c.next j))
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.homotopyCofiber.i …
  -/
  apply ext_to_X' _ _ hj
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    ι : Type u_2
    c : ComplexShape ι
    F G : HomologicalComplex C c
    φ : Quiver.Hom F G
    inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
    inst✝ : DecidableRel c.Rel
    i j : ι
    hij : c.Rel i j
    hj : Not (c.Rel j (c.next j))
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp [d_sndX φ i j hij]
  /-
    🎉 no goals
  -/


lemma shape (i j : ι) (hij : ¬ c.Rel i j) :
    d φ i j = 0 :=
  dif_neg hij


@[reassoc (attr := simp)]
lemma inrX_d (i j : ι) :
    inrX φ i ≫ d φ i j = G.d i j ≫ inrX φ j := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    ι : Type u_2
    c : ComplexShape ι
    F G : HomologicalComplex C c
    φ : Quiver.Hom F G
    inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
    inst✝ : DecidableRel c.Rel
    i j : ι
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.homotopyCofiber.i …
  -/
  by_cases hij : c.Rel i j
    /-
      case pos
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      inst✝² : CategoryTheory.Preadditive C
      ι : Type u_2
      c : ComplexShape ι
      F G : HomologicalComplex C c
      φ : Quiver.Hom F G
      inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
      inst✝ : DecidableRel c.Rel
      i j : ι
      hij : c.Rel i j
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.homotopyCofiber.i …
    -/
  · by_cases hj : c.Rel j (c.next j)
      /-
        case pos
        C : Type u_1
        inst✝³ : CategoryTheory.Category.{u_3, u_1} C
        inst✝² : CategoryTheory.Preadditive C
        ι : Type u_2
        c : ComplexShape ι
        F G : HomologicalComplex C c
        φ : Quiver.Hom F G
        inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
        inst✝ : DecidableRel c.Rel
        i j : ι
        hij : c.Rel i j
        hj : c.Rel j (c.next j)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.homotopyCofiber.i …
      -/
    · apply ext_to_X _ _ _ hj
        /-
          case pos.h₁
          C : Type u_1
          inst✝³ : CategoryTheory.Category.{u_3, u_1} C
          inst✝² : CategoryTheory.Preadditive C
          ι : Type u_2
          c : ComplexShape ι
          F G : HomologicalComplex C c
          φ : Quiver.Hom F G
          inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
          inst✝ : DecidableRel c.Rel
          i j : ι
          hij : c.Rel i j
          hj : c.Rel j (c.next j)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
      · simp [d_fstX φ _ _ _ hij]
        /-
          🎉 no goals
        -/
        /-
          case pos.h₂
          C : Type u_1
          inst✝³ : CategoryTheory.Category.{u_3, u_1} C
          inst✝² : CategoryTheory.Preadditive C
          ι : Type u_2
          c : ComplexShape ι
          F G : HomologicalComplex C c
          φ : Quiver.Hom F G
          inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
          inst✝ : DecidableRel c.Rel
          i j : ι
          hij : c.Rel i j
          hj : c.Rel j (c.next j)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
      · simp [d_sndX φ _ _ hij]
        /-
          🎉 no goals
        -/
      /-
        case neg
        C : Type u_1
        inst✝³ : CategoryTheory.Category.{u_3, u_1} C
        inst✝² : CategoryTheory.Preadditive C
        ι : Type u_2
        c : ComplexShape ι
        F G : HomologicalComplex C c
        φ : Quiver.Hom F G
        inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
        inst✝ : DecidableRel c.Rel
        i j : ι
        hij : c.Rel i j
        hj : Not (c.Rel j (c.next j))
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.homotopyCofiber.i …
      -/
    · apply ext_to_X' _ _ hj
      /-
        case neg
        C : Type u_1
        inst✝³ : CategoryTheory.Category.{u_3, u_1} C
        inst✝² : CategoryTheory.Preadditive C
        ι : Type u_2
        c : ComplexShape ι
        F G : HomologicalComplex C c
        φ : Quiver.Hom F G
        inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
        inst✝ : DecidableRel c.Rel
        i j : ι
        hij : c.Rel i j
        hj : Not (c.Rel j (c.next j))
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
      simp [d_sndX φ _ _ hij]
      /-
        🎉 no goals
      -/
    /-
      case neg
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      inst✝² : CategoryTheory.Preadditive C
      ι : Type u_2
      c : ComplexShape ι
      F G : HomologicalComplex C c
      φ : Quiver.Hom F G
      inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
      inst✝ : DecidableRel c.Rel
      i j : ι
      hij : Not (c.Rel i j)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.homotopyCofiber.i …
    -/
  · rw [shape φ _ _ hij, G.shape _ _ hij, zero_comp, comp_zero]
    /-
      🎉 no goals
    -/


/-- The homotopy cofiber of a morphism of homological complexes,
also known as the mapping cone. -/
@[simps]
noncomputable def homotopyCofiber : HomologicalComplex C c where
  X i := homotopyCofiber.X φ i
  d i j := homotopyCofiber.d φ i j
  shape i j hij := homotopyCofiber.shape φ i j hij
  d_comp_d' i j k hij hjk := by
    /-
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{?u.58977, u_1} C
      inst✝² : CategoryTheory.Preadditive C
      ι : Type u_2
      c : ComplexShape ι
      F G K : HomologicalComplex C c
      φ : Quiver.Hom F G
      inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
      inst✝ : DecidableRel c.Rel
      i j k : ι
      hij : c.Rel i j
      hjk : c.Rel j k
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun i j => HomologicalComplex.homot …
    -/
    apply homotopyCofiber.ext_from_X φ j i hij
      /-
        case h₁
        C : Type u_1
        inst✝³ : CategoryTheory.Category.{?u.58977, u_1} C
        inst✝² : CategoryTheory.Preadditive C
        ι : Type u_2
        c : ComplexShape ι
        F G K : HomologicalComplex C c
        φ : Quiver.Hom F G
        inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
        inst✝ : DecidableRel c.Rel
        i j k : ι
        hij : c.Rel i j
        hjk : c.Rel j k
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.homotopyCofiber.i …
      -/
    · dsimp
      simp only [comp_zero, homotopyCofiber.inlX_d_assoc φ i j k hij hjk,
        add_comp, assoc, homotopyCofiber.inrX_d, Hom.comm_assoc, neg_comp]
      /-
        case h₁
        C : Type u_1
        inst✝³ : CategoryTheory.Category.{?u.58977, u_1} C
        inst✝² : CategoryTheory.Preadditive C
        ι : Type u_2
        c : ComplexShape ι
        F G K : HomologicalComplex C c
        φ : Quiver.Hom F G
        inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
        inst✝ : DecidableRel c.Rel
        i j k : ι
        hij : c.Rel i j
        hjk : c.Rel j k
        ⊢ Eq (HAdd.hAdd (Neg.neg (CategoryTheory.CategoryStruct.comp (F.d j k) (Catego …
      -/
      by_cases hk : c.Rel k (c.next k)
        /-
          case pos
          C : Type u_1
          inst✝³ : CategoryTheory.Category.{?u.58977, u_1} C
          inst✝² : CategoryTheory.Preadditive C
          ι : Type u_2
          c : ComplexShape ι
          F G K : HomologicalComplex C c
          φ : Quiver.Hom F G
          inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
          inst✝ : DecidableRel c.Rel
          i j k : ι
          hij : c.Rel i j
          hjk : c.Rel j k
          hk : c.Rel k (c.next k)
          ⊢ Eq (HAdd.hAdd (Neg.neg (CategoryTheory.CategoryStruct.comp (F.d j k) (Catego …
        -/
      · simp [homotopyCofiber.inlX_d φ j k _ hjk hk]
        /-
          🎉 no goals
        -/
        /-
          case neg
          C : Type u_1
          inst✝³ : CategoryTheory.Category.{?u.58977, u_1} C
          inst✝² : CategoryTheory.Preadditive C
          ι : Type u_2
          c : ComplexShape ι
          F G K : HomologicalComplex C c
          φ : Quiver.Hom F G
          inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
          inst✝ : DecidableRel c.Rel
          i j k : ι
          hij : c.Rel i j
          hjk : c.Rel j k
          hk : Not (c.Rel k (c.next k))
          ⊢ Eq (HAdd.hAdd (Neg.neg (CategoryTheory.CategoryStruct.comp (F.d j k) (Catego …
        -/
      · simp [homotopyCofiber.inlX_d' φ j k hjk hk]
        /-
          🎉 no goals
        -/
      /-
        case h₂
        C : Type u_1
        inst✝³ : CategoryTheory.Category.{?u.58977, u_1} C
        inst✝² : CategoryTheory.Preadditive C
        ι : Type u_2
        c : ComplexShape ι
        F G K : HomologicalComplex C c
        φ : Quiver.Hom F G
        inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
        inst✝ : DecidableRel c.Rel
        i j k : ι
        hij : c.Rel i j
        hjk : c.Rel j k
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.homotopyCofiber.i …
      -/
    · simp
      /-
        🎉 no goals
      -/


/-- The right inclusion `G ⟶ homotopyCofiber φ`. -/
@[simps!]
noncomputable def inr : G ⟶ homotopyCofiber φ where
  f i := inrX φ i


/-- The composition `φ ≫ mappingCone.inr φ` is homotopic to `0`. -/
noncomputable def inrCompHomotopy (hc : ∀ j, ∃ i, c.Rel i j) :
    Homotopy (φ ≫ inr φ) 0 where
  hom i j :=
    if hij : c.Rel j i then inlX φ i j hij else 0
  zero _ _ hij := dif_neg hij
  comm j := by
    /-
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{?u.65017, u_1} C
      inst✝² : CategoryTheory.Preadditive C
      ι : Type u_2
      c : ComplexShape ι
      F G K : HomologicalComplex C c
      φ : Quiver.Hom F G
      inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
      inst✝ : DecidableRel c.Rel
      hc : ∀ (j : ι), Exists fun i => c.Rel i j
      j : ι
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp φ (HomologicalComplex.homotopyCofibe …
    -/
    obtain ⟨i, hij⟩ := hc j
    /-
      case intro
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{?u.65017, u_1} C
      inst✝² : CategoryTheory.Preadditive C
      ι : Type u_2
      c : ComplexShape ι
      F G K : HomologicalComplex C c
      φ : Quiver.Hom F G
      inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
      inst✝ : DecidableRel c.Rel
      hc : ∀ (j : ι), Exists fun i => c.Rel i j
      j i : ι
      hij : c.Rel i j
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp φ (HomologicalComplex.homotopyCofibe …
    -/
    rw [prevD_eq _ hij, dif_pos hij]
    /-
      case intro
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{?u.65017, u_1} C
      inst✝² : CategoryTheory.Preadditive C
      ι : Type u_2
      c : ComplexShape ι
      F G K : HomologicalComplex C c
      φ : Quiver.Hom F G
      inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
      inst✝ : DecidableRel c.Rel
      hc : ∀ (j : ι), Exists fun i => c.Rel i j
      j i : ι
      hij : c.Rel i j
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp φ (HomologicalComplex.homotopyCofibe …
    -/
    by_cases hj : c.Rel j (c.next j)
    · simp only [comp_f, homotopyCofiber_d, zero_f, add_zero,
        inlX_d φ i j _ hij hj, dNext_eq _ hj, dif_pos hj,
        add_neg_cancel_left, inr_f]
    · rw [dNext_eq_zero _ _  hj, zero_add, zero_f, add_zero, homotopyCofiber_d,
        inlX_d' _ _ _ _ hj, comp_f, inr_f]


lemma inrCompHomotopy_hom (i j : ι) (hij : c.Rel j i) :
    (inrCompHomotopy φ hc).hom i j = inlX φ i j hij := dif_pos hij


lemma inrCompHomotopy_hom_eq_zero (i j : ι) (hij : ¬ c.Rel j i) :
    (inrCompHomotopy φ hc).hom i j = 0 := dif_neg hij


/-- The morphism `homotopyCofiber φ ⟶ K` that is induced by a morphism `α : G ⟶ K`
and a homotopy `hα : Homotopy (φ ≫ α) 0`. -/
noncomputable def desc :
    homotopyCofiber φ ⟶ K where
  f j :=
    if hj : c.Rel j (c.next j)
    then fstX φ j _ hj ≫ hα.hom _ j + sndX φ j ≫ α.f j
    else sndX φ j ≫ α.f j
  comm' j k hjk := by
    /-
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{?u.75914, u_1} C
      inst✝² : CategoryTheory.Preadditive C
      ι : Type u_2
      c : ComplexShape ι
      F G K : HomologicalComplex C c
      φ : Quiver.Hom F G
      inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
      inst✝ : DecidableRel c.Rel
      α : Quiver.Hom G K
      hα : Homotopy (CategoryTheory.CategoryStruct.comp φ α) 0
      j k : ι
      hjk : c.Rel j k
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun j => dite (c.Rel j (c.next j))  …
    -/
    obtain rfl := c.next_eq' hjk
    /-
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{?u.75914, u_1} C
      inst✝² : CategoryTheory.Preadditive C
      ι : Type u_2
      c : ComplexShape ι
      F G K : HomologicalComplex C c
      φ : Quiver.Hom F G
      inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
      inst✝ : DecidableRel c.Rel
      α : Quiver.Hom G K
      hα : Homotopy (CategoryTheory.CategoryStruct.comp φ α) 0
      j : ι
      hjk : c.Rel j (c.next j)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun j => dite (c.Rel j (c.next j))  …
    -/
    dsimp
    /-
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{?u.75914, u_1} C
      inst✝² : CategoryTheory.Preadditive C
      ι : Type u_2
      c : ComplexShape ι
      F G K : HomologicalComplex C c
      φ : Quiver.Hom F G
      inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
      inst✝ : DecidableRel c.Rel
      α : Quiver.Hom G K
      hα : Homotopy (CategoryTheory.CategoryStruct.comp φ α) 0
      j : ι
      hjk : c.Rel j (c.next j)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (dite (c.Rel j (c.next j)) (fun hj => …
    -/
    simp [dif_pos hjk]
    /-
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{?u.75914, u_1} C
      inst✝² : CategoryTheory.Preadditive C
      ι : Type u_2
      c : ComplexShape ι
      F G K : HomologicalComplex C c
      φ : Quiver.Hom F G
      inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
      inst✝ : DecidableRel c.Rel
      α : Quiver.Hom G K
      hα : Homotopy (CategoryTheory.CategoryStruct.comp φ α) 0
      j : ι
      hjk : c.Rel j (c.next j)
      ⊢ Eq (HAdd.hAdd (CategoryTheory.CategoryStruct.comp (HomologicalComplex.homoto …
    -/
    have H := hα.comm (c.next j)
    /-
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{?u.75914, u_1} C
      inst✝² : CategoryTheory.Preadditive C
      ι : Type u_2
      c : ComplexShape ι
      F G K : HomologicalComplex C c
      φ : Quiver.Hom F G
      inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
      inst✝ : DecidableRel c.Rel
      α : Quiver.Hom G K
      hα : Homotopy (CategoryTheory.CategoryStruct.comp φ α) 0
      j : ι
      hjk : c.Rel j (c.next j)
      H : Eq ((CategoryTheory.CategoryStruct.comp φ α).f (c.next j)) (HAdd.hAdd (HAd …
      ⊢ Eq (HAdd.hAdd (CategoryTheory.CategoryStruct.comp (HomologicalComplex.homoto …
    -/
    simp only [comp_f, zero_f, add_zero, prevD_eq _ hjk] at H
    /-
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{?u.75914, u_1} C
      inst✝² : CategoryTheory.Preadditive C
      ι : Type u_2
      c : ComplexShape ι
      F G K : HomologicalComplex C c
      φ : Quiver.Hom F G
      inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
      inst✝ : DecidableRel c.Rel
      α : Quiver.Hom G K
      hα : Homotopy (CategoryTheory.CategoryStruct.comp φ α) 0
      j : ι
      hjk : c.Rel j (c.next j)
      H : Eq (CategoryTheory.CategoryStruct.comp (φ.f (c.next j)) (α.f (c.next j)))  …
      ⊢ Eq (HAdd.hAdd (CategoryTheory.CategoryStruct.comp (HomologicalComplex.homoto …
    -/
    split_ifs with hj
    · simp only [comp_add, d_sndX_assoc _ _ _ hjk, add_comp, assoc, H,
        d_fstX_assoc _ _ _ _ hjk, neg_comp, dNext, AddMonoidHom.mk'_apply]
      /-
        case pos
        C : Type u_1
        inst✝³ : CategoryTheory.Category.{?u.75914, u_1} C
        inst✝² : CategoryTheory.Preadditive C
        ι : Type u_2
        c : ComplexShape ι
        F G K : HomologicalComplex C c
        φ : Quiver.Hom F G
        inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
        inst✝ : DecidableRel c.Rel
        α : Quiver.Hom G K
        hα : Homotopy (CategoryTheory.CategoryStruct.comp φ α) 0
        j : ι
        hjk : c.Rel j (c.next j)
        H : Eq (CategoryTheory.CategoryStruct.comp (φ.f (c.next j)) (α.f (c.next j)))  …
        hj : c.Rel (c.next j) (c.next (c.next j))
        ⊢ Eq (HAdd.hAdd (CategoryTheory.CategoryStruct.comp (HomologicalComplex.homoto …
      -/
      /-
        🎉 no goals
      -/
      abel
      /-
        🎉 no goals
      -/
    · simp only [d_sndX_assoc _ _ _ hjk, add_comp, assoc, add_left_inj, H,
        dNext_eq_zero _ _ hj, zero_add]


lemma desc_f (j k : ι) (hjk : c.Rel j k) :
    (desc φ α hα).f j = fstX φ j _ hjk ≫ hα.hom _ j + sndX φ j ≫ α.f j := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    ι : Type u_2
    c : ComplexShape ι
    F G K : HomologicalComplex C c
    φ : Quiver.Hom F G
    inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
    inst✝ : DecidableRel c.Rel
    α : Quiver.Hom G K
    hα : Homotopy (CategoryTheory.CategoryStruct.comp φ α) 0
    j k : ι
    hjk : c.Rel j k
    ⊢ Eq ((HomologicalComplex.homotopyCofiber.desc φ α hα).f j) (HAdd.hAdd (Catego …
  -/
  obtain rfl := c.next_eq' hjk
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    ι : Type u_2
    c : ComplexShape ι
    F G K : HomologicalComplex C c
    φ : Quiver.Hom F G
    inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
    inst✝ : DecidableRel c.Rel
    α : Quiver.Hom G K
    hα : Homotopy (CategoryTheory.CategoryStruct.comp φ α) 0
    j : ι
    hjk : c.Rel j (c.next j)
    ⊢ Eq ((HomologicalComplex.homotopyCofiber.desc φ α hα).f j) (HAdd.hAdd (Catego …
  -/
  apply dif_pos hjk
  /-
    🎉 no goals
  -/


lemma desc_f' (j : ι) (hj : ¬ c.Rel j (c.next j)) :
    (desc φ α hα).f j = sndX φ j ≫ α.f j := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    ι : Type u_2
    c : ComplexShape ι
    F G K : HomologicalComplex C c
    φ : Quiver.Hom F G
    inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
    inst✝ : DecidableRel c.Rel
    α : Quiver.Hom G K
    hα : Homotopy (CategoryTheory.CategoryStruct.comp φ α) 0
    j : ι
    hj : Not (c.Rel j (c.next j))
    ⊢ Eq ((HomologicalComplex.homotopyCofiber.desc φ α hα).f j) (CategoryTheory.Ca …
  -/
  apply dif_neg hj
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma inlX_desc_f (i j : ι) (hjk : c.Rel j i) :
    inlX φ i j hjk ≫ (desc φ α hα).f j = hα.hom i j := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    ι : Type u_2
    c : ComplexShape ι
    F G K : HomologicalComplex C c
    φ : Quiver.Hom F G
    inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
    inst✝ : DecidableRel c.Rel
    α : Quiver.Hom G K
    hα : Homotopy (CategoryTheory.CategoryStruct.comp φ α) 0
    i j : ι
    hjk : c.Rel j i
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.homotopyCofiber.i …
  -/
  obtain rfl := c.next_eq' hjk
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    ι : Type u_2
    c : ComplexShape ι
    F G K : HomologicalComplex C c
    φ : Quiver.Hom F G
    inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
    inst✝ : DecidableRel c.Rel
    α : Quiver.Hom G K
    hα : Homotopy (CategoryTheory.CategoryStruct.comp φ α) 0
    j : ι
    hjk : c.Rel j (c.next j)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.homotopyCofiber.i …
  -/
  dsimp [desc]
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    ι : Type u_2
    c : ComplexShape ι
    F G K : HomologicalComplex C c
    φ : Quiver.Hom F G
    inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
    inst✝ : DecidableRel c.Rel
    α : Quiver.Hom G K
    hα : Homotopy (CategoryTheory.CategoryStruct.comp φ α) 0
    j : ι
    hjk : c.Rel j (c.next j)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.homotopyCofiber.i …
  -/
  rw [dif_pos hjk, comp_add, inlX_fstX_assoc, inlX_sndX_assoc, zero_comp, add_zero]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma inrX_desc_f (i : ι) :
    inrX φ i ≫ (desc φ α hα).f i = α.f i := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    ι : Type u_2
    c : ComplexShape ι
    F G K : HomologicalComplex C c
    φ : Quiver.Hom F G
    inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
    inst✝ : DecidableRel c.Rel
    α : Quiver.Hom G K
    hα : Homotopy (CategoryTheory.CategoryStruct.comp φ α) 0
    i : ι
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.homotopyCofiber.i …
  -/
  dsimp [desc]
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    ι : Type u_2
    c : ComplexShape ι
    F G K : HomologicalComplex C c
    φ : Quiver.Hom F G
    inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
    inst✝ : DecidableRel c.Rel
    α : Quiver.Hom G K
    hα : Homotopy (CategoryTheory.CategoryStruct.comp φ α) 0
    i : ι
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.homotopyCofiber.i …
  -/
                /-
                  🎉 no goals
                -/
  split_ifs <;> simp
                /-
                  🎉 no goals
                -/


@[reassoc (attr := simp)]
lemma inr_desc :
                                  /-
                                    C : Type u_1
                                    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
                                    inst✝² : CategoryTheory.Preadditive C
                                    ι : Type u_2
                                    c : ComplexShape ι
                                    F G K : HomologicalComplex C c
                                    φ : Quiver.Hom F G
                                    inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
                                    inst✝ : DecidableRel c.Rel
                                    α : Quiver.Hom G K
                                    hα : Homotopy (CategoryTheory.CategoryStruct.comp φ α) 0
                                    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.homotopyCofiber.i …
                                  -/
    inr φ ≫ desc φ α hα = α := by aesop_cat
                                  /-
                                    🎉 no goals
                                  -/


@[reassoc (attr := simp)]
lemma inrCompHomotopy_hom_desc_hom (hc : ∀ j, ∃ i, c.Rel i j) (i j : ι) :
    (inrCompHomotopy φ hc).hom i j ≫ (desc φ α hα).f j = hα.hom i j := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    ι : Type u_2
    c : ComplexShape ι
    F G K : HomologicalComplex C c
    φ : Quiver.Hom F G
    inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
    inst✝ : DecidableRel c.Rel
    α : Quiver.Hom G K
    hα : Homotopy (CategoryTheory.CategoryStruct.comp φ α) 0
    hc : ∀ (j : ι), Exists fun i => c.Rel i j
    i j : ι
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((HomologicalComplex.homotopyCofiber. …
  -/
  by_cases hij : c.Rel j i
    /-
      case pos
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      inst✝² : CategoryTheory.Preadditive C
      ι : Type u_2
      c : ComplexShape ι
      F G K : HomologicalComplex C c
      φ : Quiver.Hom F G
      inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
      inst✝ : DecidableRel c.Rel
      α : Quiver.Hom G K
      hα : Homotopy (CategoryTheory.CategoryStruct.comp φ α) 0
      hc : ∀ (j : ι), Exists fun i => c.Rel i j
      i j : ι
      hij : c.Rel j i
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((HomologicalComplex.homotopyCofiber. …
    -/
  · dsimp
    simp only [inrCompHomotopy_hom φ hc i j hij, desc_f φ α hα _ _ hij,
      comp_add, inlX_fstX_assoc, inlX_sndX_assoc, zero_comp, add_zero]
    /-
      case neg
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      inst✝² : CategoryTheory.Preadditive C
      ι : Type u_2
      c : ComplexShape ι
      F G K : HomologicalComplex C c
      φ : Quiver.Hom F G
      inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
      inst✝ : DecidableRel c.Rel
      α : Quiver.Hom G K
      hα : Homotopy (CategoryTheory.CategoryStruct.comp φ α) 0
      hc : ∀ (j : ι), Exists fun i => c.Rel i j
      i j : ι
      hij : Not (c.Rel j i)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((HomologicalComplex.homotopyCofiber. …
    -/
  · simp only [Homotopy.zero _ _ _ hij, zero_comp]
    /-
      🎉 no goals
    -/


lemma eq_desc (f : homotopyCofiber φ ⟶ K) (hc : ∀ j, ∃ i, c.Rel i j) :
                                                              /-
                                                                C : Type u_1
                                                                inst✝³ : CategoryTheory.Category.{?u.130032, u_1} C
                                                                inst✝² : CategoryTheory.Preadditive C
                                                                ι : Type u_2
                                                                c : ComplexShape ι
                                                                F G K : HomologicalComplex C c
                                                                φ : Quiver.Hom F G
                                                                inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
                                                                inst✝ : DecidableRel c.Rel
                                                                α : Quiver.Hom G K
                                                                hα : Homotopy (CategoryTheory.CategoryStruct.comp φ α) 0
                                                                f : Quiver.Hom (HomologicalComplex.homotopyCofiber φ) K
                                                                hc : ∀ (j : ι), Exists fun i => c.Rel i j
                                                                ⊢ Eq (CategoryTheory.CategoryStruct.comp φ (CategoryTheory.CategoryStruct.comp …
                                                              -/
    f = desc φ (inr φ ≫ f) (Homotopy.trans (Homotopy.ofEq (by simp))
                                                              /-
                                                                🎉 no goals
                                                              -/
                                                                     /-
                                                                       C : Type u_1
                                                                       inst✝³ : CategoryTheory.Category.{?u.130032, u_1} C
                                                                       inst✝² : CategoryTheory.Preadditive C
                                                                       ι : Type u_2
                                                                       c : ComplexShape ι
                                                                       F G K : HomologicalComplex C c
                                                                       φ : Quiver.Hom F G
                                                                       inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
                                                                       inst✝ : DecidableRel c.Rel
                                                                       α : Quiver.Hom G K
                                                                       hα : Homotopy (CategoryTheory.CategoryStruct.comp φ α) 0
                                                                       f : Quiver.Hom (HomologicalComplex.homotopyCofiber φ) K
                                                                       hc : ∀ (j : ι), Exists fun i => c.Rel i j
                                                                       ⊢ Eq (CategoryTheory.CategoryStruct.comp 0 f) 0
                                                                     -/
      (((inrCompHomotopy φ hc).compRight f).trans (Homotopy.ofEq (by simp)))) := by
                                                                     /-
                                                                       🎉 no goals
                                                                     -/
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    ι : Type u_2
    c : ComplexShape ι
    F G K : HomologicalComplex C c
    φ : Quiver.Hom F G
    inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
    inst✝ : DecidableRel c.Rel
    f : Quiver.Hom (HomologicalComplex.homotopyCofiber φ) K
    hc : ∀ (j : ι), Exists fun i => c.Rel i j
    ⊢ Eq f (HomologicalComplex.homotopyCofiber.desc φ (CategoryTheory.CategoryStru …
  -/
  ext j
  /-
    case h
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    ι : Type u_2
    c : ComplexShape ι
    F G K : HomologicalComplex C c
    φ : Quiver.Hom F G
    inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
    inst✝ : DecidableRel c.Rel
    f : Quiver.Hom (HomologicalComplex.homotopyCofiber φ) K
    hc : ∀ (j : ι), Exists fun i => c.Rel i j
    j : ι
    ⊢ Eq (f.f j) ((HomologicalComplex.homotopyCofiber.desc φ (CategoryTheory.Categ …
  -/
  by_cases hj : c.Rel j (c.next j)
    /-
      case pos
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      inst✝² : CategoryTheory.Preadditive C
      ι : Type u_2
      c : ComplexShape ι
      F G K : HomologicalComplex C c
      φ : Quiver.Hom F G
      inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
      inst✝ : DecidableRel c.Rel
      f : Quiver.Hom (HomologicalComplex.homotopyCofiber φ) K
      hc : ∀ (j : ι), Exists fun i => c.Rel i j
      j : ι
      hj : c.Rel j (c.next j)
      ⊢ Eq (f.f j) ((HomologicalComplex.homotopyCofiber.desc φ (CategoryTheory.Categ …
    -/
  · apply ext_from_X φ _ _ hj
      /-
        case pos.h₁
        C : Type u_1
        inst✝³ : CategoryTheory.Category.{u_3, u_1} C
        inst✝² : CategoryTheory.Preadditive C
        ι : Type u_2
        c : ComplexShape ι
        F G K : HomologicalComplex C c
        φ : Quiver.Hom F G
        inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
        inst✝ : DecidableRel c.Rel
        f : Quiver.Hom (HomologicalComplex.homotopyCofiber φ) K
        hc : ∀ (j : ι), Exists fun i => c.Rel i j
        j : ι
        hj : c.Rel j (c.next j)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.homotopyCofiber.i …
      -/
    · simp [inrCompHomotopy_hom _ _ _ _ hj]
      /-
        🎉 no goals
      -/
      /-
        case pos.h₂
        C : Type u_1
        inst✝³ : CategoryTheory.Category.{u_3, u_1} C
        inst✝² : CategoryTheory.Preadditive C
        ι : Type u_2
        c : ComplexShape ι
        F G K : HomologicalComplex C c
        φ : Quiver.Hom F G
        inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
        inst✝ : DecidableRel c.Rel
        f : Quiver.Hom (HomologicalComplex.homotopyCofiber φ) K
        hc : ∀ (j : ι), Exists fun i => c.Rel i j
        j : ι
        hj : c.Rel j (c.next j)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.homotopyCofiber.i …
      -/
    · simp
      /-
        🎉 no goals
      -/
    /-
      case neg
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      inst✝² : CategoryTheory.Preadditive C
      ι : Type u_2
      c : ComplexShape ι
      F G K : HomologicalComplex C c
      φ : Quiver.Hom F G
      inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
      inst✝ : DecidableRel c.Rel
      f : Quiver.Hom (HomologicalComplex.homotopyCofiber φ) K
      hc : ∀ (j : ι), Exists fun i => c.Rel i j
      j : ι
      hj : Not (c.Rel j (c.next j))
      ⊢ Eq (f.f j) ((HomologicalComplex.homotopyCofiber.desc φ (CategoryTheory.Categ …
    -/
  · apply ext_from_X' φ _ hj
    /-
      case neg
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      inst✝² : CategoryTheory.Preadditive C
      ι : Type u_2
      c : ComplexShape ι
      F G K : HomologicalComplex C c
      φ : Quiver.Hom F G
      inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
      inst✝ : DecidableRel c.Rel
      f : Quiver.Hom (HomologicalComplex.homotopyCofiber φ) K
      hc : ∀ (j : ι), Exists fun i => c.Rel i j
      j : ι
      hj : Not (c.Rel j (c.next j))
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.homotopyCofiber.i …
    -/
    simp
    /-
      🎉 no goals
    -/


lemma descSigma_ext_iff {φ : F ⟶ G} {K : HomologicalComplex C c}
    (x y : Σ (α : G ⟶ K), Homotopy (φ ≫ α) 0) :
    x = y ↔ x.1 = y.1 ∧ (∀ (i j : ι) (_ : c.Rel j i), x.2.hom i j = y.2.hom i j) := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    ι : Type u_2
    c : ComplexShape ι
    F G : HomologicalComplex C c
    inst✝ : DecidableRel c.Rel
    φ : Quiver.Hom F G
    K : HomologicalComplex C c
    x y : Sigma fun α => Homotopy (CategoryTheory.CategoryStruct.comp φ α) 0
    ⊢ Iff (Eq x y) (And (Eq x.fst y.fst) (∀ (i j : ι), c.Rel j i → Eq (x.snd.hom i …
  -/
  constructor
    /-
      case mp
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      ι : Type u_2
      c : ComplexShape ι
      F G : HomologicalComplex C c
      inst✝ : DecidableRel c.Rel
      φ : Quiver.Hom F G
      K : HomologicalComplex C c
      x y : Sigma fun α => Homotopy (CategoryTheory.CategoryStruct.comp φ α) 0
      ⊢ Eq x y → And (Eq x.fst y.fst) (∀ (i j : ι), c.Rel j i → Eq (x.snd.hom i j) ( …
    -/
  · rintro rfl
    /-
      case mp
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      ι : Type u_2
      c : ComplexShape ι
      F G : HomologicalComplex C c
      inst✝ : DecidableRel c.Rel
      φ : Quiver.Hom F G
      K : HomologicalComplex C c
      x : Sigma fun α => Homotopy (CategoryTheory.CategoryStruct.comp φ α) 0
      ⊢ And (Eq x.fst x.fst) (∀ (i j : ι), c.Rel j i → Eq (x.snd.hom i j) (x.snd.hom …
    -/
    tauto
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      ι : Type u_2
      c : ComplexShape ι
      F G : HomologicalComplex C c
      inst✝ : DecidableRel c.Rel
      φ : Quiver.Hom F G
      K : HomologicalComplex C c
      x y : Sigma fun α => Homotopy (CategoryTheory.CategoryStruct.comp φ α) 0
      ⊢ And (Eq x.fst y.fst) (∀ (i j : ι), c.Rel j i → Eq (x.snd.hom i j) (y.snd.hom …
    -/
  · obtain ⟨x₁, x₂⟩ := x
    /-
      case mpr.mk
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      ι : Type u_2
      c : ComplexShape ι
      F G : HomologicalComplex C c
      inst✝ : DecidableRel c.Rel
      φ : Quiver.Hom F G
      K : HomologicalComplex C c
      y : Sigma fun α => Homotopy (CategoryTheory.CategoryStruct.comp φ α) 0
      x₁ : Quiver.Hom G K
      x₂ : Homotopy (CategoryTheory.CategoryStruct.comp φ x₁) 0
      ⊢ And (Eq ⟨x₁, x₂⟩.fst y.fst) (∀ (i j : ι), c.Rel j i → Eq (⟨x₁, x₂⟩.snd.hom i …
    -/
    obtain ⟨y₁, y₂⟩ := y
    /-
      case mpr.mk.mk
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      ι : Type u_2
      c : ComplexShape ι
      F G : HomologicalComplex C c
      inst✝ : DecidableRel c.Rel
      φ : Quiver.Hom F G
      K : HomologicalComplex C c
      x₁ : Quiver.Hom G K
      x₂ : Homotopy (CategoryTheory.CategoryStruct.comp φ x₁) 0
      y₁ : Quiver.Hom G K
      y₂ : Homotopy (CategoryTheory.CategoryStruct.comp φ y₁) 0
      ⊢ And (Eq ⟨x₁, x₂⟩.fst ⟨y₁, y₂⟩.fst) (∀ (i j : ι), c.Rel j i → Eq (⟨x₁, x₂⟩.sn …
    -/
    rintro ⟨rfl, h⟩
    /-
      case mpr.mk.mk.intro
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      ι : Type u_2
      c : ComplexShape ι
      F G : HomologicalComplex C c
      inst✝ : DecidableRel c.Rel
      φ : Quiver.Hom F G
      K : HomologicalComplex C c
      x₁ : Quiver.Hom G K
      x₂ : Homotopy (CategoryTheory.CategoryStruct.comp φ x₁) 0
      y₂ : Homotopy (CategoryTheory.CategoryStruct.comp φ ⟨x₁, x₂⟩.fst) 0
      h : ∀ (i j : ι), c.Rel j i → Eq (⟨x₁, x₂⟩.snd.hom i j) (⟨⟨x₁, x₂⟩.fst, y₂⟩.snd …
      ⊢ Eq ⟨x₁, x₂⟩ ⟨⟨x₁, x₂⟩.fst, y₂⟩
    -/
    simp only [Sigma.mk.inj_iff, heq_eq_eq, true_and]
    /-
      case mpr.mk.mk.intro
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      ι : Type u_2
      c : ComplexShape ι
      F G : HomologicalComplex C c
      inst✝ : DecidableRel c.Rel
      φ : Quiver.Hom F G
      K : HomologicalComplex C c
      x₁ : Quiver.Hom G K
      x₂ : Homotopy (CategoryTheory.CategoryStruct.comp φ x₁) 0
      y₂ : Homotopy (CategoryTheory.CategoryStruct.comp φ ⟨x₁, x₂⟩.fst) 0
      h : ∀ (i j : ι), c.Rel j i → Eq (⟨x₁, x₂⟩.snd.hom i j) (⟨⟨x₁, x₂⟩.fst, y₂⟩.snd …
      ⊢ Eq x₂ y₂
    -/
    ext i j
    /-
      case mpr.mk.mk.intro.hom.h.h
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      ι : Type u_2
      c : ComplexShape ι
      F G : HomologicalComplex C c
      inst✝ : DecidableRel c.Rel
      φ : Quiver.Hom F G
      K : HomologicalComplex C c
      x₁ : Quiver.Hom G K
      x₂ : Homotopy (CategoryTheory.CategoryStruct.comp φ x₁) 0
      y₂ : Homotopy (CategoryTheory.CategoryStruct.comp φ ⟨x₁, x₂⟩.fst) 0
      h : ∀ (i j : ι), c.Rel j i → Eq (⟨x₁, x₂⟩.snd.hom i j) (⟨⟨x₁, x₂⟩.fst, y₂⟩.snd …
      i j : ι
      ⊢ Eq (x₂.hom i j) (y₂.hom i j)
    -/
    by_cases hij : c.Rel j i
      /-
        case pos
        C : Type u_1
        inst✝² : CategoryTheory.Category.{u_3, u_1} C
        inst✝¹ : CategoryTheory.Preadditive C
        ι : Type u_2
        c : ComplexShape ι
        F G : HomologicalComplex C c
        inst✝ : DecidableRel c.Rel
        φ : Quiver.Hom F G
        K : HomologicalComplex C c
        x₁ : Quiver.Hom G K
        x₂ : Homotopy (CategoryTheory.CategoryStruct.comp φ x₁) 0
        y₂ : Homotopy (CategoryTheory.CategoryStruct.comp φ ⟨x₁, x₂⟩.fst) 0
        h : ∀ (i j : ι), c.Rel j i → Eq (⟨x₁, x₂⟩.snd.hom i j) (⟨⟨x₁, x₂⟩.fst, y₂⟩.snd …
        i j : ι
        hij : c.Rel j i
        ⊢ Eq (x₂.hom i j) (y₂.hom i j)
      -/
    · exact h _ _ hij
      /-
        🎉 no goals
      -/
      /-
        case neg
        C : Type u_1
        inst✝² : CategoryTheory.Category.{u_3, u_1} C
        inst✝¹ : CategoryTheory.Preadditive C
        ι : Type u_2
        c : ComplexShape ι
        F G : HomologicalComplex C c
        inst✝ : DecidableRel c.Rel
        φ : Quiver.Hom F G
        K : HomologicalComplex C c
        x₁ : Quiver.Hom G K
        x₂ : Homotopy (CategoryTheory.CategoryStruct.comp φ x₁) 0
        y₂ : Homotopy (CategoryTheory.CategoryStruct.comp φ ⟨x₁, x₂⟩.fst) 0
        h : ∀ (i j : ι), c.Rel j i → Eq (⟨x₁, x₂⟩.snd.hom i j) (⟨⟨x₁, x₂⟩.fst, y₂⟩.snd …
        i j : ι
        hij : Not (c.Rel j i)
        ⊢ Eq (x₂.hom i j) (y₂.hom i j)
      -/
    · simp only [Homotopy.zero _ _ _ hij]
      /-
        🎉 no goals
      -/


/-- Morphisms `homotopyCofiber φ ⟶ K` are uniquely determined by
a morphism `α : G ⟶ K` and a homotopy from `φ ≫ α` to `0`. -/
noncomputable def descEquiv (K : HomologicalComplex C c) (hc : ∀ j, ∃ i, c.Rel i j) :
    (Σ (α : G ⟶ K), Homotopy (φ ≫ α) 0) ≃ (homotopyCofiber φ ⟶ K) where
  toFun := fun ⟨α, hα⟩ => desc φ α hα
                                                            /-
                                                              C : Type u_1
                                                              inst✝³ : CategoryTheory.Category.{?u.142101, u_1} C
                                                              inst✝² : CategoryTheory.Preadditive C
                                                              ι : Type u_2
                                                              c : ComplexShape ι
                                                              F G K✝ : HomologicalComplex C c
                                                              φ : Quiver.Hom F G
                                                              inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
                                                              inst✝ : DecidableRel c.Rel
                                                              K : HomologicalComplex C c
                                                              hc : ∀ (j : ι), Exists fun i => c.Rel i j
                                                              f : Quiver.Hom (HomologicalComplex.homotopyCofiber φ) K
                                                              ⊢ Eq (CategoryTheory.CategoryStruct.comp φ (CategoryTheory.CategoryStruct.comp …
                                                            -/
  invFun f := ⟨inr φ ≫ f, Homotopy.trans (Homotopy.ofEq (by simp))
                                                            /-
                                                              🎉 no goals
                                                            -/
                                                                   /-
                                                                     C : Type u_1
                                                                     inst✝³ : CategoryTheory.Category.{?u.142101, u_1} C
                                                                     inst✝² : CategoryTheory.Preadditive C
                                                                     ι : Type u_2
                                                                     c : ComplexShape ι
                                                                     F G K✝ : HomologicalComplex C c
                                                                     φ : Quiver.Hom F G
                                                                     inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
                                                                     inst✝ : DecidableRel c.Rel
                                                                     K : HomologicalComplex C c
                                                                     hc : ∀ (j : ι), Exists fun i => c.Rel i j
                                                                     f : Quiver.Hom (HomologicalComplex.homotopyCofiber φ) K
                                                                     ⊢ Eq (CategoryTheory.CategoryStruct.comp 0 f) 0
                                                                   -/
    (((inrCompHomotopy φ hc).compRight f).trans (Homotopy.ofEq (by simp)))⟩
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
  right_inv f := (eq_desc φ f hc).symm
  left_inv := fun ⟨α, hα⟩ => by
    /-
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{?u.142101, u_1} C
      inst✝² : CategoryTheory.Preadditive C
      ι : Type u_2
      c : ComplexShape ι
      F G K✝ : HomologicalComplex C c
      φ : Quiver.Hom F G
      inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
      inst✝ : DecidableRel c.Rel
      K : HomologicalComplex C c
      hc : ∀ (j : ι), Exists fun i => c.Rel i j
      x✝ : Sigma fun α => Homotopy (CategoryTheory.CategoryStruct.comp φ α) 0
      α : Quiver.Hom G K
      hα : Homotopy (CategoryTheory.CategoryStruct.comp φ α) 0
      ⊢ Eq ((fun f => ⟨CategoryTheory.CategoryStruct.comp (HomologicalComplex.homoto …
    -/
    rw [descSigma_ext_iff]
    /-
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{?u.142101, u_1} C
      inst✝² : CategoryTheory.Preadditive C
      ι : Type u_2
      c : ComplexShape ι
      F G K✝ : HomologicalComplex C c
      φ : Quiver.Hom F G
      inst✝¹ : HomologicalComplex.HasHomotopyCofiber φ
      inst✝ : DecidableRel c.Rel
      K : HomologicalComplex C c
      hc : ∀ (j : ι), Exists fun i => c.Rel i j
      x✝ : Sigma fun α => Homotopy (CategoryTheory.CategoryStruct.comp φ α) 0
      α : Quiver.Hom G K
      hα : Homotopy (CategoryTheory.CategoryStruct.comp φ α) 0
      ⊢ And (Eq ((fun f => ⟨CategoryTheory.CategoryStruct.comp (HomologicalComplex.h …
    -/
    aesop_cat
    /-
      🎉 no goals
    -/


/-- The cylinder object of a homological complex `K` is the homotopy cofiber
of the morphism  `biprod.lift (𝟙 K) (-𝟙 K) : K ⟶ K ⊞ K`. -/
noncomputable abbrev cylinder := homotopyCofiber (biprod.lift (𝟙 K) (-𝟙 K))


/-- The left inclusion `K ⟶ K.cylinder`. -/
noncomputable def ι₀ : K ⟶ K.cylinder := biprod.inl ≫ homotopyCofiber.inr _


/-- The right inclusion `K ⟶ K.cylinder`. -/
noncomputable def ι₁ : K ⟶ K.cylinder := biprod.inr ≫ homotopyCofiber.inr _


/-- The morphism `K.cylinder ⟶ F` that is induced by two morphisms `φ₀ φ₁ : K ⟶ F`
and a homotopy `h : Homotopy φ₀ φ₁`. -/
noncomputable def desc : K.cylinder ⟶ F :=
  homotopyCofiber.desc _ (biprod.desc φ₀ φ₁)
    (Homotopy.trans (Homotopy.ofEq (by
      /-
        C : Type u_1
        inst✝⁵ : CategoryTheory.Category.{?u.168959, u_1} C
        inst✝⁴ : CategoryTheory.Preadditive C
        ι : Type u_2
        c : ComplexShape ι
        F G K : HomologicalComplex C c
        φ : Quiver.Hom F G
        inst✝³ : HomologicalComplex.HasHomotopyCofiber φ
        inst✝² : DecidableRel c.Rel
        inst✝¹ : ∀ (i : ι), CategoryTheory.Limits.HasBinaryBiproduct (K.X i) (K.X i)
        inst✝ : HomologicalComplex.HasHomotopyCofiber (CategoryTheory.Limits.biprod.li …
        φ₀ φ₁ : Quiver.Hom K F
        h : Homotopy φ₀ φ₁
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.biprod.lift (C …
      -/
      simp only [biprod.lift_desc, id_comp, neg_comp, sub_eq_add_neg]))
      /-
        🎉 no goals
      -/
      ((Homotopy.equivSubZero h)))


@[reassoc (attr := simp)]
                                               /-
                                                 C : Type u_1
                                                 inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
                                                 inst✝³ : CategoryTheory.Preadditive C
                                                 ι : Type u_2
                                                 c : ComplexShape ι
                                                 F K : HomologicalComplex C c
                                                 inst✝² : DecidableRel c.Rel
                                                 inst✝¹ : ∀ (i : ι), CategoryTheory.Limits.HasBinaryBiproduct (K.X i) (K.X i)
                                                 inst✝ : HomologicalComplex.HasHomotopyCofiber (CategoryTheory.Limits.biprod.li …
                                                 φ₀ φ₁ : Quiver.Hom K F
                                                 h : Homotopy φ₀ φ₁
                                                 ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.cylinder.ι₀ K) (H …
                                               -/
lemma ι₀_desc : ι₀ K ≫ desc φ₀ φ₁ h = φ₀ := by simp [ι₀, desc]
                                               /-
                                                 🎉 no goals
                                               -/


@[reassoc (attr := simp)]
                                               /-
                                                 C : Type u_1
                                                 inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
                                                 inst✝³ : CategoryTheory.Preadditive C
                                                 ι : Type u_2
                                                 c : ComplexShape ι
                                                 F K : HomologicalComplex C c
                                                 inst✝² : DecidableRel c.Rel
                                                 inst✝¹ : ∀ (i : ι), CategoryTheory.Limits.HasBinaryBiproduct (K.X i) (K.X i)
                                                 inst✝ : HomologicalComplex.HasHomotopyCofiber (CategoryTheory.Limits.biprod.li …
                                                 φ₀ φ₁ : Quiver.Hom K F
                                                 h : Homotopy φ₀ φ₁
                                                 ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.cylinder.ι₁ K) (H …
                                               -/
lemma ι₁_desc : ι₁ K ≫ desc φ₀ φ₁ h = φ₁ := by simp [ι₁, desc]
                                               /-
                                                 🎉 no goals
                                               -/


/-- The projection `π : K.cylinder ⟶ K`. -/
noncomputable def π : K.cylinder ⟶ K := desc (𝟙 K) (𝟙 K) (Homotopy.refl _)


@[reassoc (attr := simp)]
                                    /-
                                      C : Type u_1
                                      inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
                                      inst✝³ : CategoryTheory.Preadditive C
                                      ι : Type u_2
                                      c : ComplexShape ι
                                      K : HomologicalComplex C c
                                      inst✝² : DecidableRel c.Rel
                                      inst✝¹ : ∀ (i : ι), CategoryTheory.Limits.HasBinaryBiproduct (K.X i) (K.X i)
                                      inst✝ : HomologicalComplex.HasHomotopyCofiber (CategoryTheory.Limits.biprod.li …
                                      ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.cylinder.ι₀ K) (H …
                                    -/
lemma ι₀_π : ι₀ K ≫ π K = 𝟙 K := by simp [π]
                                    /-
                                      🎉 no goals
                                    -/


@[reassoc (attr := simp)]
                                    /-
                                      C : Type u_1
                                      inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
                                      inst✝³ : CategoryTheory.Preadditive C
                                      ι : Type u_2
                                      c : ComplexShape ι
                                      K : HomologicalComplex C c
                                      inst✝² : DecidableRel c.Rel
                                      inst✝¹ : ∀ (i : ι), CategoryTheory.Limits.HasBinaryBiproduct (K.X i) (K.X i)
                                      inst✝ : HomologicalComplex.HasHomotopyCofiber (CategoryTheory.Limits.biprod.li …
                                      ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.cylinder.ι₁ K) (H …
                                    -/
lemma ι₁_π : ι₁ K ≫ π K = 𝟙 K := by simp [π]
                                    /-
                                      🎉 no goals
                                    -/


/-- The left inclusion `K.X i ⟶ K.cylinder.X j` when `c.Rel j i`. -/
noncomputable abbrev inlX (i j : ι) (hij : c.Rel j i) : K.X i ⟶ K.cylinder.X j :=
  homotopyCofiber.inlX (biprod.lift (𝟙 K) (-𝟙 K)) i j hij


/-- The right inclusion `(K ⊞ K).X i ⟶ K.cylinder.X i`. -/
noncomputable abbrev inrX (i : ι) : (K ⊞ K).X i ⟶ K.cylinder.X i :=
  homotopyCofiber.inrX (biprod.lift (𝟙 K) (-𝟙 K)) i


@[reassoc (attr := simp)]
lemma inlX_π (i j : ι) (hij : c.Rel j i) :
    inlX K i j hij ≫ (π K).f j = 0 := by
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
    inst✝³ : CategoryTheory.Preadditive C
    ι : Type u_2
    c : ComplexShape ι
    K : HomologicalComplex C c
    inst✝² : DecidableRel c.Rel
    inst✝¹ : ∀ (i : ι), CategoryTheory.Limits.HasBinaryBiproduct (K.X i) (K.X i)
    inst✝ : HomologicalComplex.HasHomotopyCofiber (CategoryTheory.Limits.biprod.li …
    i j : ι
    hij : c.Rel j i
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.cylinder.inlX K i …
  -/
  erw [homotopyCofiber.inlX_desc_f]
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
    inst✝³ : CategoryTheory.Preadditive C
    ι : Type u_2
    c : ComplexShape ι
    K : HomologicalComplex C c
    inst✝² : DecidableRel c.Rel
    inst✝¹ : ∀ (i : ι), CategoryTheory.Limits.HasBinaryBiproduct (K.X i) (K.X i)
    inst✝ : HomologicalComplex.HasHomotopyCofiber (CategoryTheory.Limits.biprod.li …
    i j : ι
    hij : c.Rel j i
    ⊢ Eq (((Homotopy.ofEq ⋯).trans (Homotopy.equivSubZero (Homotopy.refl (Category …
  -/
  simp [Homotopy.equivSubZero]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma inrX_π (i : ι) :
    inrX K i ≫ (π K).f i = (biprod.desc (𝟙 _) (𝟙 K)).f i :=
  homotopyCofiber.inrX_desc_f _ _ _ _


/-- A null homotopic map `K.cylinder ⟶ K.cylinder` which identifies to
`π K ≫ ι₀ K - 𝟙 _`, see `nullHomotopicMap_eq`. -/
noncomputable def nullHomotopicMap : K.cylinder ⟶ K.cylinder :=
  Homotopy.nullHomotopicMap'
    (fun i j hij => homotopyCofiber.sndX (biprod.lift (𝟙 K) (-𝟙 K)) i ≫
      (biprod.snd : K ⊞ K ⟶ K).f i ≫ inlX K i j hij)


/-- The obvious homotopy from `nullHomotopicMap K` to zero. -/
noncomputable def nullHomotopy : Homotopy (nullHomotopicMap K) 0 :=
  Homotopy.nullHomotopy' _


lemma inlX_nullHomotopy_f (i j : ι) (hij : c.Rel j i) :
    inlX K i j hij ≫ (nullHomotopicMap K).f j =
      inlX K i j hij ≫ (π K ≫ ι₀ K - 𝟙 _).f j := by
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
    inst✝³ : CategoryTheory.Preadditive C
    ι : Type u_2
    c : ComplexShape ι
    K : HomologicalComplex C c
    inst✝² : DecidableRel c.Rel
    inst✝¹ : ∀ (i : ι), CategoryTheory.Limits.HasBinaryBiproduct (K.X i) (K.X i)
    inst✝ : HomologicalComplex.HasHomotopyCofiber (CategoryTheory.Limits.biprod.li …
    i j : ι
    hij : c.Rel j i
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.cylinder.inlX K i …
  -/
  dsimp [nullHomotopicMap]
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
    inst✝³ : CategoryTheory.Preadditive C
    ι : Type u_2
    c : ComplexShape ι
    K : HomologicalComplex C c
    inst✝² : DecidableRel c.Rel
    inst✝¹ : ∀ (i : ι), CategoryTheory.Limits.HasBinaryBiproduct (K.X i) (K.X i)
    inst✝ : HomologicalComplex.HasHomotopyCofiber (CategoryTheory.Limits.biprod.li …
    i j : ι
    hij : c.Rel j i
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.cylinder.inlX K i …
  -/
  by_cases hj : ∃ (k : ι), c.Rel k j
    /-
      case pos
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
      inst✝³ : CategoryTheory.Preadditive C
      ι : Type u_2
      c : ComplexShape ι
      K : HomologicalComplex C c
      inst✝² : DecidableRel c.Rel
      inst✝¹ : ∀ (i : ι), CategoryTheory.Limits.HasBinaryBiproduct (K.X i) (K.X i)
      inst✝ : HomologicalComplex.HasHomotopyCofiber (CategoryTheory.Limits.biprod.li …
      i j : ι
      hij : c.Rel j i
      hj : Exists fun k => c.Rel k j
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.cylinder.inlX K i …
    -/
  · obtain ⟨k, hjk⟩ := hj
    simp only [assoc, Homotopy.nullHomotopicMap'_f hjk hij, homotopyCofiber_X, homotopyCofiber_d,
      homotopyCofiber.d_sndX_assoc _ _ _ hij, add_comp, comp_add, homotopyCofiber.inlX_fstX_assoc,
      homotopyCofiber.inlX_sndX_assoc, zero_comp, add_zero, comp_sub, inlX_π_assoc, comp_id,
      zero_sub, ← HomologicalComplex.comp_f_assoc, biprod.lift_snd, neg_f_apply, id_f,
      neg_comp, id_comp]
    /-
      case neg
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
      inst✝³ : CategoryTheory.Preadditive C
      ι : Type u_2
      c : ComplexShape ι
      K : HomologicalComplex C c
      inst✝² : DecidableRel c.Rel
      inst✝¹ : ∀ (i : ι), CategoryTheory.Limits.HasBinaryBiproduct (K.X i) (K.X i)
      inst✝ : HomologicalComplex.HasHomotopyCofiber (CategoryTheory.Limits.biprod.li …
      i j : ι
      hij : c.Rel j i
      hj : Not (Exists fun k => c.Rel k j)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.cylinder.inlX K i …
    -/
  · simp only [not_exists] at hj
    simp only [Homotopy.nullHomotopicMap'_f_of_not_rel_right hij hj,
      homotopyCofiber_X, homotopyCofiber_d, assoc, comp_sub, comp_id,
      homotopyCofiber.d_sndX_assoc _ _ _ hij, add_comp, comp_add, zero_comp, add_zero,
      homotopyCofiber.inlX_fstX_assoc, homotopyCofiber.inlX_sndX_assoc,
      ← HomologicalComplex.comp_f_assoc, biprod.lift_snd, neg_f_apply, id_f, neg_comp,
      id_comp, inlX_π_assoc, zero_sub]


lemma inrX_nullHomotopy_f (j : ι) :
    inrX K j ≫ (nullHomotopicMap K).f j = inrX K j ≫ (π K ≫ ι₀ K - 𝟙 _).f j := by
  have : biprod.lift (𝟙 K) (-𝟙 K) = biprod.inl - biprod.inr :=
    biprod.hom_ext _ _ (by simp) (by simp)
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
    inst✝³ : CategoryTheory.Preadditive C
    ι : Type u_2
    c : ComplexShape ι
    K : HomologicalComplex C c
    inst✝² : DecidableRel c.Rel
    inst✝¹ : ∀ (i : ι), CategoryTheory.Limits.HasBinaryBiproduct (K.X i) (K.X i)
    inst✝ : HomologicalComplex.HasHomotopyCofiber (CategoryTheory.Limits.biprod.li …
    hc : ∀ (j : ι), Exists fun i => c.Rel i j
    j : ι
    this : Eq (CategoryTheory.Limits.biprod.lift (CategoryTheory.CategoryStruct.id …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.cylinder.inrX K j …
  -/
  obtain ⟨i, hij⟩ := hc j
  /-
    case intro
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
    inst✝³ : CategoryTheory.Preadditive C
    ι : Type u_2
    c : ComplexShape ι
    K : HomologicalComplex C c
    inst✝² : DecidableRel c.Rel
    inst✝¹ : ∀ (i : ι), CategoryTheory.Limits.HasBinaryBiproduct (K.X i) (K.X i)
    inst✝ : HomologicalComplex.HasHomotopyCofiber (CategoryTheory.Limits.biprod.li …
    hc : ∀ (j : ι), Exists fun i => c.Rel i j
    j : ι
    this : Eq (CategoryTheory.Limits.biprod.lift (CategoryTheory.CategoryStruct.id …
    i : ι
    hij : c.Rel i j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.cylinder.inrX K j …
  -/
  dsimp [nullHomotopicMap]
  /-
    case intro
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
    inst✝³ : CategoryTheory.Preadditive C
    ι : Type u_2
    c : ComplexShape ι
    K : HomologicalComplex C c
    inst✝² : DecidableRel c.Rel
    inst✝¹ : ∀ (i : ι), CategoryTheory.Limits.HasBinaryBiproduct (K.X i) (K.X i)
    inst✝ : HomologicalComplex.HasHomotopyCofiber (CategoryTheory.Limits.biprod.li …
    hc : ∀ (j : ι), Exists fun i => c.Rel i j
    j : ι
    this : Eq (CategoryTheory.Limits.biprod.lift (CategoryTheory.CategoryStruct.id …
    i : ι
    hij : c.Rel i j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.cylinder.inrX K j …
  -/
  by_cases hj : ∃ (k : ι), c.Rel j k
    /-
      case pos
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
      inst✝³ : CategoryTheory.Preadditive C
      ι : Type u_2
      c : ComplexShape ι
      K : HomologicalComplex C c
      inst✝² : DecidableRel c.Rel
      inst✝¹ : ∀ (i : ι), CategoryTheory.Limits.HasBinaryBiproduct (K.X i) (K.X i)
      inst✝ : HomologicalComplex.HasHomotopyCofiber (CategoryTheory.Limits.biprod.li …
      hc : ∀ (j : ι), Exists fun i => c.Rel i j
      j : ι
      this : Eq (CategoryTheory.Limits.biprod.lift (CategoryTheory.CategoryStruct.id …
      i : ι
      hij : c.Rel i j
      hj : Exists fun k => c.Rel j k
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.cylinder.inrX K j …
    -/
  · obtain ⟨k, hjk⟩ := hj
    simp only [Homotopy.nullHomotopicMap'_f hij hjk,
      homotopyCofiber_X, homotopyCofiber_d, assoc, comp_add,
      homotopyCofiber.inrX_d_assoc, homotopyCofiber.inrX_sndX_assoc, comp_sub,
      inrX_π_assoc, comp_id, ← Hom.comm_assoc, homotopyCofiber.inlX_d _ _ _ _ _ hjk,
      comp_neg, add_neg_cancel_left]
    /-
      case pos.intro
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
      inst✝³ : CategoryTheory.Preadditive C
      ι : Type u_2
      c : ComplexShape ι
      K : HomologicalComplex C c
      inst✝² : DecidableRel c.Rel
      inst✝¹ : ∀ (i : ι), CategoryTheory.Limits.HasBinaryBiproduct (K.X i) (K.X i)
      inst✝ : HomologicalComplex.HasHomotopyCofiber (CategoryTheory.Limits.biprod.li …
      hc : ∀ (j : ι), Exists fun i => c.Rel i j
      j : ι
      this : Eq (CategoryTheory.Limits.biprod.lift (CategoryTheory.CategoryStruct.id …
      i : ι
      hij : c.Rel i j
      k : ι
      hjk : c.Rel j k
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.biprod.snd.f j …
    -/
    rw [← cancel_epi (biprodXIso K K j).inv]
    /-
      case pos.intro
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
      inst✝³ : CategoryTheory.Preadditive C
      ι : Type u_2
      c : ComplexShape ι
      K : HomologicalComplex C c
      inst✝² : DecidableRel c.Rel
      inst✝¹ : ∀ (i : ι), CategoryTheory.Limits.HasBinaryBiproduct (K.X i) (K.X i)
      inst✝ : HomologicalComplex.HasHomotopyCofiber (CategoryTheory.Limits.biprod.li …
      hc : ∀ (j : ι), Exists fun i => c.Rel i j
      j : ι
      this : Eq (CategoryTheory.Limits.biprod.lift (CategoryTheory.CategoryStruct.id …
      i : ι
      hij : c.Rel i j
      k : ι
      hjk : c.Rel j k
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.biprodXIso K j).inv (CategoryTheor …
    -/
    ext
      /-
        case pos.intro.h₀
        C : Type u_1
        inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
        inst✝³ : CategoryTheory.Preadditive C
        ι : Type u_2
        c : ComplexShape ι
        K : HomologicalComplex C c
        inst✝² : DecidableRel c.Rel
        inst✝¹ : ∀ (i : ι), CategoryTheory.Limits.HasBinaryBiproduct (K.X i) (K.X i)
        inst✝ : HomologicalComplex.HasHomotopyCofiber (CategoryTheory.Limits.biprod.li …
        hc : ∀ (j : ι), Exists fun i => c.Rel i j
        j : ι
        this : Eq (CategoryTheory.Limits.biprod.lift (CategoryTheory.CategoryStruct.id …
        i : ι
        hij : c.Rel i j
        k : ι
        hjk : c.Rel j k
        ⊢ Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.Limits.biprod.inl (Cat …
      -/
    · simp [ι₀]
      /-
        🎉 no goals
      -/
      /-
        case pos.intro.h₁
        C : Type u_1
        inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
        inst✝³ : CategoryTheory.Preadditive C
        ι : Type u_2
        c : ComplexShape ι
        K : HomologicalComplex C c
        inst✝² : DecidableRel c.Rel
        inst✝¹ : ∀ (i : ι), CategoryTheory.Limits.HasBinaryBiproduct (K.X i) (K.X i)
        inst✝ : HomologicalComplex.HasHomotopyCofiber (CategoryTheory.Limits.biprod.li …
        hc : ∀ (j : ι), Exists fun i => c.Rel i j
        j : ι
        this : Eq (CategoryTheory.Limits.biprod.lift (CategoryTheory.CategoryStruct.id …
        i : ι
        hij : c.Rel i j
        k : ι
        hjk : c.Rel j k
        ⊢ Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.Limits.biprod.inr (Cat …
      -/
    · dsimp
      simp only [inr_biprodXIso_inv_assoc, biprod_inr_snd_f_assoc, comp_sub,
        biprod_inr_desc_f_assoc, id_f, id_comp, ι₀, comp_f, this,
        sub_f_apply, sub_comp, homotopyCofiber_X, homotopyCofiber.inr_f]
    /-
      case neg
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
      inst✝³ : CategoryTheory.Preadditive C
      ι : Type u_2
      c : ComplexShape ι
      K : HomologicalComplex C c
      inst✝² : DecidableRel c.Rel
      inst✝¹ : ∀ (i : ι), CategoryTheory.Limits.HasBinaryBiproduct (K.X i) (K.X i)
      inst✝ : HomologicalComplex.HasHomotopyCofiber (CategoryTheory.Limits.biprod.li …
      hc : ∀ (j : ι), Exists fun i => c.Rel i j
      j : ι
      this : Eq (CategoryTheory.Limits.biprod.lift (CategoryTheory.CategoryStruct.id …
      i : ι
      hij : c.Rel i j
      hj : Not (Exists fun k => c.Rel j k)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.cylinder.inrX K j …
    -/
  · simp only [not_exists] at hj
    simp only [assoc, Homotopy.nullHomotopicMap'_f_of_not_rel_left hij hj, homotopyCofiber_X,
      homotopyCofiber_d, homotopyCofiber.inlX_d' _ _ _ _ (hj _), homotopyCofiber.inrX_sndX_assoc,
      comp_sub, inrX_π_assoc, comp_id, ι₀, comp_f, homotopyCofiber.inr_f]
    /-
      case neg
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
      inst✝³ : CategoryTheory.Preadditive C
      ι : Type u_2
      c : ComplexShape ι
      K : HomologicalComplex C c
      inst✝² : DecidableRel c.Rel
      inst✝¹ : ∀ (i : ι), CategoryTheory.Limits.HasBinaryBiproduct (K.X i) (K.X i)
      inst✝ : HomologicalComplex.HasHomotopyCofiber (CategoryTheory.Limits.biprod.li …
      hc : ∀ (j : ι), Exists fun i => c.Rel i j
      j : ι
      this : Eq (CategoryTheory.Limits.biprod.lift (CategoryTheory.CategoryStruct.id …
      i : ι
      hij : c.Rel i j
      hj : ∀ (x : ι), Not (c.Rel j x)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.biprod.snd.f j …
    -/
    rw [← cancel_epi (biprodXIso K K j).inv]
    /-
      case neg
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
      inst✝³ : CategoryTheory.Preadditive C
      ι : Type u_2
      c : ComplexShape ι
      K : HomologicalComplex C c
      inst✝² : DecidableRel c.Rel
      inst✝¹ : ∀ (i : ι), CategoryTheory.Limits.HasBinaryBiproduct (K.X i) (K.X i)
      inst✝ : HomologicalComplex.HasHomotopyCofiber (CategoryTheory.Limits.biprod.li …
      hc : ∀ (j : ι), Exists fun i => c.Rel i j
      j : ι
      this : Eq (CategoryTheory.Limits.biprod.lift (CategoryTheory.CategoryStruct.id …
      i : ι
      hij : c.Rel i j
      hj : ∀ (x : ι), Not (c.Rel j x)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.biprodXIso K j).inv (CategoryTheor …
    -/
    ext
      /-
        case neg.h₀
        C : Type u_1
        inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
        inst✝³ : CategoryTheory.Preadditive C
        ι : Type u_2
        c : ComplexShape ι
        K : HomologicalComplex C c
        inst✝² : DecidableRel c.Rel
        inst✝¹ : ∀ (i : ι), CategoryTheory.Limits.HasBinaryBiproduct (K.X i) (K.X i)
        inst✝ : HomologicalComplex.HasHomotopyCofiber (CategoryTheory.Limits.biprod.li …
        hc : ∀ (j : ι), Exists fun i => c.Rel i j
        j : ι
        this : Eq (CategoryTheory.Limits.biprod.lift (CategoryTheory.CategoryStruct.id …
        i : ι
        hij : c.Rel i j
        hj : ∀ (x : ι), Not (c.Rel j x)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.Limits.biprod.inl (Cat …
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case neg.h₁
        C : Type u_1
        inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
        inst✝³ : CategoryTheory.Preadditive C
        ι : Type u_2
        c : ComplexShape ι
        K : HomologicalComplex C c
        inst✝² : DecidableRel c.Rel
        inst✝¹ : ∀ (i : ι), CategoryTheory.Limits.HasBinaryBiproduct (K.X i) (K.X i)
        inst✝ : HomologicalComplex.HasHomotopyCofiber (CategoryTheory.Limits.biprod.li …
        hc : ∀ (j : ι), Exists fun i => c.Rel i j
        j : ι
        this : Eq (CategoryTheory.Limits.biprod.lift (CategoryTheory.CategoryStruct.id …
        i : ι
        hij : c.Rel i j
        hj : ∀ (x : ι), Not (c.Rel j x)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.Limits.biprod.inr (Cat …
      -/
    · simp [this]
      /-
        🎉 no goals
      -/


lemma nullHomotopicMap_eq : nullHomotopicMap K = π K ≫ ι₀ K - 𝟙 _ := by
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
    inst✝³ : CategoryTheory.Preadditive C
    ι : Type u_2
    c : ComplexShape ι
    K : HomologicalComplex C c
    inst✝² : DecidableRel c.Rel
    inst✝¹ : ∀ (i : ι), CategoryTheory.Limits.HasBinaryBiproduct (K.X i) (K.X i)
    inst✝ : HomologicalComplex.HasHomotopyCofiber (CategoryTheory.Limits.biprod.li …
    hc : ∀ (j : ι), Exists fun i => c.Rel i j
    ⊢ Eq (HomologicalComplex.cylinder.πCompι₀Homotopy.nullHomotopicMap K) (HSub.hS …
  -/
  ext i
  /-
    case h
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
    inst✝³ : CategoryTheory.Preadditive C
    ι : Type u_2
    c : ComplexShape ι
    K : HomologicalComplex C c
    inst✝² : DecidableRel c.Rel
    inst✝¹ : ∀ (i : ι), CategoryTheory.Limits.HasBinaryBiproduct (K.X i) (K.X i)
    inst✝ : HomologicalComplex.HasHomotopyCofiber (CategoryTheory.Limits.biprod.li …
    hc : ∀ (j : ι), Exists fun i => c.Rel i j
    i : ι
    ⊢ Eq ((HomologicalComplex.cylinder.πCompι₀Homotopy.nullHomotopicMap K).f i) (( …
  -/
  by_cases hi : c.Rel i (c.next i)
  · exact homotopyCofiber.ext_from_X (biprod.lift (𝟙 K) (-𝟙 K)) (c.next i) i hi
      (inlX_nullHomotopy_f _ _ _ _) (inrX_nullHomotopy_f _ hc _)
    /-
      case neg
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
      inst✝³ : CategoryTheory.Preadditive C
      ι : Type u_2
      c : ComplexShape ι
      K : HomologicalComplex C c
      inst✝² : DecidableRel c.Rel
      inst✝¹ : ∀ (i : ι), CategoryTheory.Limits.HasBinaryBiproduct (K.X i) (K.X i)
      inst✝ : HomologicalComplex.HasHomotopyCofiber (CategoryTheory.Limits.biprod.li …
      hc : ∀ (j : ι), Exists fun i => c.Rel i j
      i : ι
      hi : Not (c.Rel i (c.next i))
      ⊢ Eq ((HomologicalComplex.cylinder.πCompι₀Homotopy.nullHomotopicMap K).f i) (( …
    -/
  · exact homotopyCofiber.ext_from_X' (biprod.lift (𝟙 K) (-𝟙 K)) _ hi (inrX_nullHomotopy_f _ hc _)
    /-
      🎉 no goals
    -/


/-- The homotopy between `π K ≫ ι₀ K` and `𝟙 K.cylinder`. -/
noncomputable def πCompι₀Homotopy : Homotopy (π K ≫ ι₀ K) (𝟙 K.cylinder) :=
  Homotopy.equivSubZero.symm
    ((Homotopy.ofEq (πCompι₀Homotopy.nullHomotopicMap_eq K hc).symm).trans
      (πCompι₀Homotopy.nullHomotopy K))


/-- The homotopy equivalence between `K.cylinder` and `K`. -/
noncomputable def homotopyEquiv : HomotopyEquiv K.cylinder K where
  hom := π K
  inv := ι₀ K
  homotopyHomInvId := πCompι₀Homotopy K hc
                                        /-
                                          C : Type u_1
                                          inst✝⁵ : CategoryTheory.Category.{?u.299701, u_1} C
                                          inst✝⁴ : CategoryTheory.Preadditive C
                                          ι : Type u_2
                                          c : ComplexShape ι
                                          F G K : HomologicalComplex C c
                                          φ : Quiver.Hom F G
                                          inst✝³ : HomologicalComplex.HasHomotopyCofiber φ
                                          inst✝² : DecidableRel c.Rel
                                          inst✝¹ : ∀ (i : ι), CategoryTheory.Limits.HasBinaryBiproduct (K.X i) (K.X i)
                                          inst✝ : HomologicalComplex.HasHomotopyCofiber (CategoryTheory.Limits.biprod.li …
                                          hc : ∀ (j : ι), Exists fun i => c.Rel i j
                                          ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.cylinder.ι₀ K) (H …
                                        -/
  homotopyInvHomId := Homotopy.ofEq (by simp)
                                        /-
                                          🎉 no goals
                                        -/


/-- The homotopy between `cylinder.ι₀ K` and `cylinder.ι₁ K`. -/
noncomputable def homotopy₀₁ : Homotopy (ι₀ K) (ι₁ K) :=
                     /-
                       C : Type u_1
                       inst✝⁵ : CategoryTheory.Category.{?u.302274, u_1} C
                       inst✝⁴ : CategoryTheory.Preadditive C
                       ι : Type u_2
                       c : ComplexShape ι
                       F G K : HomologicalComplex C c
                       φ : Quiver.Hom F G
                       inst✝³ : HomologicalComplex.HasHomotopyCofiber φ
                       inst✝² : DecidableRel c.Rel
                       inst✝¹ : ∀ (i : ι), CategoryTheory.Limits.HasBinaryBiproduct (K.X i) (K.X i)
                       inst✝ : HomologicalComplex.HasHomotopyCofiber (CategoryTheory.Limits.biprod.li …
                       hc : ∀ (j : ι), Exists fun i => c.Rel i j
                       ⊢ Eq (HomologicalComplex.cylinder.ι₀ K) (CategoryTheory.CategoryStruct.comp (H …
                     -/
  (Homotopy.ofEq (by simp)).trans (((πCompι₀Homotopy K hc).compLeft (ι₁ K)).trans
                     /-
                       🎉 no goals
                     -/
                       /-
                         C : Type u_1
                         inst✝⁵ : CategoryTheory.Category.{?u.302274, u_1} C
                         inst✝⁴ : CategoryTheory.Preadditive C
                         ι : Type u_2
                         c : ComplexShape ι
                         F G K : HomologicalComplex C c
                         φ : Quiver.Hom F G
                         inst✝³ : HomologicalComplex.HasHomotopyCofiber φ
                         inst✝² : DecidableRel c.Rel
                         inst✝¹ : ∀ (i : ι), CategoryTheory.Limits.HasBinaryBiproduct (K.X i) (K.X i)
                         inst✝ : HomologicalComplex.HasHomotopyCofiber (CategoryTheory.Limits.biprod.li …
                         hc : ∀ (j : ι), Exists fun i => c.Rel i j
                         ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.cylinder.ι₁ K) (C …
                       -/
    (Homotopy.ofEq (by simp)))
                       /-
                         🎉 no goals
                       -/


include hc in
lemma map_ι₀_eq_map_ι₁ {D : Type*} [Category D] (H : HomologicalComplex C c ⥤ D)
    (hH : (homotopyEquivalences C c).IsInvertedBy H) :
    H.map (ι₀ K) = H.map (ι₁ K) := by
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_5, u_1} C
    inst✝⁴ : CategoryTheory.Preadditive C
    ι : Type u_2
    c : ComplexShape ι
    K : HomologicalComplex C c
    inst✝³ : DecidableRel c.Rel
    inst✝² : ∀ (i : ι), CategoryTheory.Limits.HasBinaryBiproduct (K.X i) (K.X i)
    inst✝¹ : HomologicalComplex.HasHomotopyCofiber (CategoryTheory.Limits.biprod.l …
    hc : ∀ (j : ι), Exists fun i => c.Rel i j
    D : Type u_3
    inst✝ : CategoryTheory.Category.{u_4, u_3} D
    H : CategoryTheory.Functor (HomologicalComplex C c) D
    hH : (HomologicalComplex.homotopyEquivalences C c).IsInvertedBy H
    ⊢ Eq (H.map (HomologicalComplex.cylinder.ι₀ K)) (H.map (HomologicalComplex.cyl …
  -/
  have : IsIso (H.map (cylinder.π K)) := hH _ ⟨homotopyEquiv K hc, rfl⟩
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_5, u_1} C
    inst✝⁴ : CategoryTheory.Preadditive C
    ι : Type u_2
    c : ComplexShape ι
    K : HomologicalComplex C c
    inst✝³ : DecidableRel c.Rel
    inst✝² : ∀ (i : ι), CategoryTheory.Limits.HasBinaryBiproduct (K.X i) (K.X i)
    inst✝¹ : HomologicalComplex.HasHomotopyCofiber (CategoryTheory.Limits.biprod.l …
    hc : ∀ (j : ι), Exists fun i => c.Rel i j
    D : Type u_3
    inst✝ : CategoryTheory.Category.{u_4, u_3} D
    H : CategoryTheory.Functor (HomologicalComplex C c) D
    hH : (HomologicalComplex.homotopyEquivalences C c).IsInvertedBy H
    this : CategoryTheory.IsIso (H.map (HomologicalComplex.cylinder.π K))
    ⊢ Eq (H.map (HomologicalComplex.cylinder.ι₀ K)) (H.map (HomologicalComplex.cyl …
  -/
  simp only [← cancel_mono (H.map (cylinder.π K)), ← H.map_comp, ι₀_π, H.map_id, ι₁_π]
  /-
    🎉 no goals
  -/


/-- If a functor inverts homotopy equivalences, it sends homotopic maps to the same map. -/
lemma _root_.Homotopy.map_eq_of_inverts_homotopyEquivalences
    {φ₀ φ₁ : F ⟶ G} (h : Homotopy φ₀ φ₁)(hc : ∀ j, ∃ i, c.Rel i j)
    [∀ i, HasBinaryBiproduct (F.X i) (F.X i)]
    [HasHomotopyCofiber (biprod.lift (𝟙 F) (-𝟙 F))]
    {D : Type*} [Category D] (H : HomologicalComplex C c ⥤ D)
    (hH : (homotopyEquivalences C c).IsInvertedBy H) :
    H.map φ₀ = H.map φ₁ := by
  simp only [← cylinder.ι₀_desc _ _ h, ← cylinder.ι₁_desc _ _ h, H.map_comp,
    cylinder.map_ι₀_eq_map_ι₁ _ hc _ hH]


