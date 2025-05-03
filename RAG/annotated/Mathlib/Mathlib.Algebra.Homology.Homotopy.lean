/-- The composition of `C.d i (c.next i) ≫ f (c.next i) i`. -/
def dNext (i : ι) : (∀ i j, C.X i ⟶ D.X j) →+ (C.X i ⟶ D.X i) :=
  AddMonoidHom.mk' (fun f => C.d i (c.next i) ≫ f (c.next i) i) fun _ _ =>
    Preadditive.comp_add _ _ _ _ _ _


/-- `f (c.next i) i`. -/
def fromNext (i : ι) : (∀ i j, C.X i ⟶ D.X j) →+ (C.xNext i ⟶ D.X i) :=
  AddMonoidHom.mk' (fun f => f (c.next i) i) fun _ _ => rfl


@[simp]
theorem dNext_eq_dFrom_fromNext (f : ∀ i j, C.X i ⟶ D.X j) (i : ι) :
    dNext i f = C.dFrom i ≫ fromNext i f :=
  rfl


theorem dNext_eq (f : ∀ i j, C.X i ⟶ D.X j) {i i' : ι} (w : c.Rel i i') :
    dNext i f = C.d i i' ≫ f i' i := by
  /-
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    c : ComplexShape ι
    C D : HomologicalComplex V c
    f : (i j : ι) → Quiver.Hom (C.X i) (D.X j)
    i i' : ι
    w : c.Rel i i'
    ⊢ Eq ((dNext i) f) (CategoryTheory.CategoryStruct.comp (C.d i i') (f i' i))
  -/
  obtain rfl := c.next_eq' w
  /-
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    c : ComplexShape ι
    C D : HomologicalComplex V c
    f : (i j : ι) → Quiver.Hom (C.X i) (D.X j)
    i : ι
    w : c.Rel i (c.next i)
    ⊢ Eq ((dNext i) f) (CategoryTheory.CategoryStruct.comp (C.d i (c.next i)) (f ( …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma dNext_eq_zero (f : ∀ i j, C.X i ⟶ D.X j) (i : ι) (hi : ¬ c.Rel i (c.next i)) :
    dNext i f = 0 := by
  /-
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    c : ComplexShape ι
    C D : HomologicalComplex V c
    f : (i j : ι) → Quiver.Hom (C.X i) (D.X j)
    i : ι
    hi : Not (c.Rel i (c.next i))
    ⊢ Eq ((dNext i) f) 0
  -/
  dsimp [dNext]
  /-
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    c : ComplexShape ι
    C D : HomologicalComplex V c
    f : (i j : ι) → Quiver.Hom (C.X i) (D.X j)
    i : ι
    hi : Not (c.Rel i (c.next i))
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (C.d i (c.next i)) (f (c.next i) i)) 0
  -/
  rw [shape _ _ _ hi, zero_comp]
  /-
    🎉 no goals
  -/


@[simp 1100]
theorem dNext_comp_left (f : C ⟶ D) (g : ∀ i j, D.X i ⟶ E.X j) (i : ι) :
    (dNext i fun i j => f.f i ≫ g i j) = f.f i ≫ dNext i g :=
  (f.comm_assoc _ _ _).symm


@[simp 1100]
theorem dNext_comp_right (f : ∀ i j, C.X i ⟶ D.X j) (g : D ⟶ E) (i : ι) :
    (dNext i fun i j => f i j ≫ g.f j) = dNext i f ≫ g.f i :=
  (assoc _ _ _).symm


/-- The composition `f j (c.prev j) ≫ D.d (c.prev j) j`. -/
def prevD (j : ι) : (∀ i j, C.X i ⟶ D.X j) →+ (C.X j ⟶ D.X j) :=
  AddMonoidHom.mk' (fun f => f j (c.prev j) ≫ D.d (c.prev j) j) fun _ _ =>
    Preadditive.add_comp _ _ _ _ _ _


lemma prevD_eq_zero (f : ∀ i j, C.X i ⟶ D.X j) (i : ι) (hi : ¬ c.Rel (c.prev i) i) :
    prevD i f = 0 := by
  /-
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    c : ComplexShape ι
    C D : HomologicalComplex V c
    f : (i j : ι) → Quiver.Hom (C.X i) (D.X j)
    i : ι
    hi : Not (c.Rel (c.prev i) i)
    ⊢ Eq ((prevD i) f) 0
  -/
  dsimp [prevD]
  /-
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    c : ComplexShape ι
    C D : HomologicalComplex V c
    f : (i j : ι) → Quiver.Hom (C.X i) (D.X j)
    i : ι
    hi : Not (c.Rel (c.prev i) i)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (f i (c.prev i)) (D.d (c.prev i) i)) 0
  -/
  rw [shape _ _ _ hi, comp_zero]
  /-
    🎉 no goals
  -/


/-- `f j (c.prev j)`. -/
def toPrev (j : ι) : (∀ i j, C.X i ⟶ D.X j) →+ (C.X j ⟶ D.xPrev j) :=
  AddMonoidHom.mk' (fun f => f j (c.prev j)) fun _ _ => rfl


@[simp]
theorem prevD_eq_toPrev_dTo (f : ∀ i j, C.X i ⟶ D.X j) (j : ι) :
    prevD j f = toPrev j f ≫ D.dTo j :=
  rfl


theorem prevD_eq (f : ∀ i j, C.X i ⟶ D.X j) {j j' : ι} (w : c.Rel j' j) :
    prevD j f = f j j' ≫ D.d j' j := by
  /-
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    c : ComplexShape ι
    C D : HomologicalComplex V c
    f : (i j : ι) → Quiver.Hom (C.X i) (D.X j)
    j j' : ι
    w : c.Rel j' j
    ⊢ Eq ((prevD j) f) (CategoryTheory.CategoryStruct.comp (f j j') (D.d j' j))
  -/
  obtain rfl := c.prev_eq' w
  /-
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    c : ComplexShape ι
    C D : HomologicalComplex V c
    f : (i j : ι) → Quiver.Hom (C.X i) (D.X j)
    j : ι
    w : c.Rel (c.prev j) j
    ⊢ Eq ((prevD j) f) (CategoryTheory.CategoryStruct.comp (f j (c.prev j)) (D.d ( …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp 1100]
theorem prevD_comp_left (f : C ⟶ D) (g : ∀ i j, D.X i ⟶ E.X j) (j : ι) :
    (prevD j fun i j => f.f i ≫ g i j) = f.f j ≫ prevD j g :=
  assoc _ _ _


@[simp 1100]
theorem prevD_comp_right (f : ∀ i j, C.X i ⟶ D.X j) (g : D ⟶ E) (j : ι) :
    (prevD j fun i j => f i j ≫ g.f j) = prevD j f ≫ g.f j := by
  /-
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    c : ComplexShape ι
    C D E : HomologicalComplex V c
    f : (i j : ι) → Quiver.Hom (C.X i) (D.X j)
    g : Quiver.Hom D E
    j : ι
    ⊢ Eq ((prevD j) fun i j => CategoryTheory.CategoryStruct.comp (f i j) (g.f j)) …
  -/
  dsimp [prevD]
  /-
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    c : ComplexShape ι
    C D E : HomologicalComplex V c
    f : (i j : ι) → Quiver.Hom (C.X i) (D.X j)
    g : Quiver.Hom D E
    j : ι
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp only [assoc, g.comm]
  /-
    🎉 no goals
  -/


theorem dNext_nat (C D : ChainComplex V ℕ) (i : ℕ) (f : ∀ i j, C.X i ⟶ D.X j) :
    dNext i f = C.d i (i - 1) ≫ f (i - 1) i := by
  /-
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    C D : ChainComplex V Nat
    i : Nat
    f : (i j : Nat) → Quiver.Hom (C.X i) (D.X j)
    ⊢ Eq ((dNext i) f) (CategoryTheory.CategoryStruct.comp (C.d i (HSub.hSub i 1)) …
  -/
  dsimp [dNext]
  /-
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    C D : ChainComplex V Nat
    i : Nat
    f : (i j : Nat) → Quiver.Hom (C.X i) (D.X j)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (C.d i ((ComplexShape.down Nat).next  …
  -/
  cases i
  · simp only [shape, ChainComplex.next_nat_zero, ComplexShape.down_Rel, Nat.one_ne_zero,
      not_false_iff, zero_comp, reduceCtorEq]
    /-
      case succ
      V : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} V
      inst✝ : CategoryTheory.Preadditive V
      C D : ChainComplex V Nat
      f : (i j : Nat) → Quiver.Hom (C.X i) (D.X j)
      n✝ : Nat
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (C.d (HAdd.hAdd n✝ 1) ((ComplexShape. …
    -/
              /-
                🎉 no goals
              -/
              /-
                🎉 no goals
              -/
  · congr <;> simp
              /-
                🎉 no goals
              -/


theorem prevD_nat (C D : CochainComplex V ℕ) (i : ℕ) (f : ∀ i j, C.X i ⟶ D.X j) :
    prevD i f = f i (i - 1) ≫ D.d (i - 1) i := by
  /-
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    C D : CochainComplex V Nat
    i : Nat
    f : (i j : Nat) → Quiver.Hom (C.X i) (D.X j)
    ⊢ Eq ((prevD i) f) (CategoryTheory.CategoryStruct.comp (f i (HSub.hSub i 1)) ( …
  -/
  dsimp [prevD]
  /-
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    C D : CochainComplex V Nat
    i : Nat
    f : (i j : Nat) → Quiver.Hom (C.X i) (D.X j)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (f i ((ComplexShape.up Nat).prev i))  …
  -/
  cases i
  · simp only [shape, CochainComplex.prev_nat_zero, ComplexShape.up_Rel, Nat.one_ne_zero,
      not_false_iff, comp_zero, reduceCtorEq]
    /-
      case succ
      V : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} V
      inst✝ : CategoryTheory.Preadditive V
      C D : CochainComplex V Nat
      f : (i j : Nat) → Quiver.Hom (C.X i) (D.X j)
      n✝ : Nat
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (f (HAdd.hAdd n✝ 1) ((ComplexShape.up …
    -/
              /-
                🎉 no goals
              -/
              /-
                🎉 no goals
              -/
  · congr <;> simp
              /-
                🎉 no goals
              -/

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): removed @[has_nonempty_instance]

/-- A homotopy `h` between chain maps `f` and `g` consists of components `h i j : C.X i ⟶ D.X j`
which are zero unless `c.Rel j i`, satisfying the homotopy condition.
-/
@[ext]
structure Homotopy (f g : C ⟶ D) where
  hom : ∀ i j, C.X i ⟶ D.X j
  zero : ∀ i j, ¬c.Rel j i → hom i j = 0 := by aesop_cat
  comm : ∀ i, f.f i = dNext i hom + prevD i hom + g.f i := by aesop_cat


/-- `f` is homotopic to `g` iff `f - g` is homotopic to `0`.
-/
def equivSubZero : Homotopy f g ≃ Homotopy (f - g) 0 where
  toFun h :=
    { hom := fun i j => h.hom i j
      zero := fun _ _ w => h.zero _ _ w
                          /-
                            ι : Type u_1
                            V : Type u
                            inst✝¹ : CategoryTheory.Category.{v, u} V
                            inst✝ : CategoryTheory.Preadditive V
                            c : ComplexShape ι
                            C D E : HomologicalComplex V c
                            f g : Quiver.Hom C D
                            h✝ k : Quiver.Hom D E
                            i✝ : ι
                            h : Homotopy f g
                            i : ι
                            ⊢ Eq ((HSub.hSub f g).f i) (HAdd.hAdd (HAdd.hAdd ((dNext i) fun i j => h.hom i …
                          -/
      comm := fun i => by simp [h.comm] }
                          /-
                            🎉 no goals
                          -/
  invFun h :=
    { hom := fun i j => h.hom i j
      zero := fun _ _ w => h.zero _ _ w
                          /-
                            ι : Type u_1
                            V : Type u
                            inst✝¹ : CategoryTheory.Category.{v, u} V
                            inst✝ : CategoryTheory.Preadditive V
                            c : ComplexShape ι
                            C D E : HomologicalComplex V c
                            f g : Quiver.Hom C D
                            h✝ k : Quiver.Hom D E
                            i✝ : ι
                            h : Homotopy (HSub.hSub f g) 0
                            i : ι
                            ⊢ Eq (f.f i) (HAdd.hAdd (HAdd.hAdd ((dNext i) fun i j => h.hom i j) ((prevD i) …
                          -/
      comm := fun i => by simpa [sub_eq_iff_eq_add] using h.comm i }
                          /-
                            🎉 no goals
                          -/
                 /-
                   ι : Type u_1
                   V : Type u
                   inst✝¹ : CategoryTheory.Category.{v, u} V
                   inst✝ : CategoryTheory.Preadditive V
                   c : ComplexShape ι
                   C D E : HomologicalComplex V c
                   f g : Quiver.Hom C D
                   h k : Quiver.Hom D E
                   i : ι
                   ⊢ Function.LeftInverse (fun h => { hom := fun i j => h.hom i j, zero := ⋯, com …
                 -/
  left_inv := by aesop_cat
                 /-
                   🎉 no goals
                 -/
                  /-
                    ι : Type u_1
                    V : Type u
                    inst✝¹ : CategoryTheory.Category.{v, u} V
                    inst✝ : CategoryTheory.Preadditive V
                    c : ComplexShape ι
                    C D E : HomologicalComplex V c
                    f g : Quiver.Hom C D
                    h k : Quiver.Hom D E
                    i : ι
                    ⊢ Function.RightInverse (fun h => { hom := fun i j => h.hom i j, zero := ⋯, co …
                  -/
  right_inv := by aesop_cat
                  /-
                    🎉 no goals
                  -/


/-- Equal chain maps are homotopic. -/
@[simps]
def ofEq (h : f = g) : Homotopy f g where
  hom := 0
  zero _ _ _ := rfl


/-- Every chain map is homotopic to itself. -/
@[simps!, refl]
def refl (f : C ⟶ D) : Homotopy f f :=
  ofEq (rfl : f = f)


/-- `f` is homotopic to `g` iff `g` is homotopic to `f`. -/
@[simps!, symm]
def symm {f g : C ⟶ D} (h : Homotopy f g) : Homotopy g f where
  hom := -h.hom
                   /-
                     ι : Type u_1
                     V : Type u
                     inst✝¹ : CategoryTheory.Category.{v, u} V
                     inst✝ : CategoryTheory.Preadditive V
                     c : ComplexShape ι
                     C D E : HomologicalComplex V c
                     f✝ g✝ : Quiver.Hom C D
                     h✝ k : Quiver.Hom D E
                     i✝ : ι
                     f g : Quiver.Hom C D
                     h : Homotopy f g
                     i j : ι
                     w : Not (c.Rel j i)
                     ⊢ Eq (Neg.neg h.hom i j) 0
                   -/
  zero i j w := by rw [Pi.neg_apply, Pi.neg_apply, h.zero i j w, neg_zero]
                   /-
                     🎉 no goals
                   -/
  comm i := by
    rw [AddMonoidHom.map_neg, AddMonoidHom.map_neg, h.comm, ← neg_add, ← add_assoc, neg_add_cancel,
      zero_add]


/-- homotopy is a transitive relation. -/
@[simps!, trans]
def trans {e f g : C ⟶ D} (h : Homotopy e f) (k : Homotopy f g) : Homotopy e g where
  hom := h.hom + k.hom
                   /-
                     ι : Type u_1
                     V : Type u
                     inst✝¹ : CategoryTheory.Category.{v, u} V
                     inst✝ : CategoryTheory.Preadditive V
                     c : ComplexShape ι
                     C D E : HomologicalComplex V c
                     f✝ g✝ : Quiver.Hom C D
                     h✝ k✝ : Quiver.Hom D E
                     i✝ : ι
                     e f g : Quiver.Hom C D
                     h : Homotopy e f
                     k : Homotopy f g
                     i j : ι
                     w : Not (c.Rel j i)
                     ⊢ Eq (HAdd.hAdd h.hom k.hom i j) 0
                   -/
  zero i j w := by rw [Pi.add_apply, Pi.add_apply, h.zero i j w, k.zero i j w, zero_add]
                   /-
                     🎉 no goals
                   -/
  comm i := by
    /-
      ι : Type u_1
      V : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} V
      inst✝ : CategoryTheory.Preadditive V
      c : ComplexShape ι
      C D E : HomologicalComplex V c
      f✝ g✝ : Quiver.Hom C D
      h✝ k✝ : Quiver.Hom D E
      i✝ : ι
      e f g : Quiver.Hom C D
      h : Homotopy e f
      k : Homotopy f g
      i : ι
      ⊢ Eq (e.f i) (HAdd.hAdd (HAdd.hAdd ((dNext i) (HAdd.hAdd h.hom k.hom)) ((prevD …
    -/
    rw [AddMonoidHom.map_add, AddMonoidHom.map_add, h.comm, k.comm]
    /-
      ι : Type u_1
      V : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} V
      inst✝ : CategoryTheory.Preadditive V
      c : ComplexShape ι
      C D E : HomologicalComplex V c
      f✝ g✝ : Quiver.Hom C D
      h✝ k✝ : Quiver.Hom D E
      i✝ : ι
      e f g : Quiver.Hom C D
      h : Homotopy e f
      k : Homotopy f g
      i : ι
      ⊢ Eq (HAdd.hAdd (HAdd.hAdd ((dNext i) h.hom) ((prevD i) h.hom)) (HAdd.hAdd (HA …
    -/
    /-
      🎉 no goals
    -/
    abel
    /-
      🎉 no goals
    -/


/-- the sum of two homotopies is a homotopy between the sum of the respective morphisms. -/
@[simps!]
def add {f₁ g₁ f₂ g₂ : C ⟶ D} (h₁ : Homotopy f₁ g₁) (h₂ : Homotopy f₂ g₂) :
    Homotopy (f₁ + f₂) (g₁ + g₂) where
  hom := h₁.hom + h₂.hom
                     /-
                       ι : Type u_1
                       V : Type u
                       inst✝¹ : CategoryTheory.Category.{v, u} V
                       inst✝ : CategoryTheory.Preadditive V
                       c : ComplexShape ι
                       C D E : HomologicalComplex V c
                       f g : Quiver.Hom C D
                       h k : Quiver.Hom D E
                       i✝ : ι
                       f₁ g₁ f₂ g₂ : Quiver.Hom C D
                       h₁ : Homotopy f₁ g₁
                       h₂ : Homotopy f₂ g₂
                       i j : ι
                       hij : Not (c.Rel j i)
                       ⊢ Eq (HAdd.hAdd h₁.hom h₂.hom i j) 0
                     -/
  zero i j hij := by rw [Pi.add_apply, Pi.add_apply, h₁.zero i j hij, h₂.zero i j hij, add_zero]
                     /-
                       🎉 no goals
                     -/
  comm i := by
    /-
      ι : Type u_1
      V : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} V
      inst✝ : CategoryTheory.Preadditive V
      c : ComplexShape ι
      C D E : HomologicalComplex V c
      f g : Quiver.Hom C D
      h k : Quiver.Hom D E
      i✝ : ι
      f₁ g₁ f₂ g₂ : Quiver.Hom C D
      h₁ : Homotopy f₁ g₁
      h₂ : Homotopy f₂ g₂
      i : ι
      ⊢ Eq ((HAdd.hAdd f₁ f₂).f i) (HAdd.hAdd (HAdd.hAdd ((dNext i) (HAdd.hAdd h₁.ho …
    -/
    simp only [HomologicalComplex.add_f_apply, h₁.comm, h₂.comm, AddMonoidHom.map_add]
    /-
      ι : Type u_1
      V : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} V
      inst✝ : CategoryTheory.Preadditive V
      c : ComplexShape ι
      C D E : HomologicalComplex V c
      f g : Quiver.Hom C D
      h k : Quiver.Hom D E
      i✝ : ι
      f₁ g₁ f₂ g₂ : Quiver.Hom C D
      h₁ : Homotopy f₁ g₁
      h₂ : Homotopy f₂ g₂
      i : ι
      ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd ((dNext i) h₁.hom) ((prevD i) h₁.hom)) ( …
    -/
    /-
      🎉 no goals
    -/
    abel
    /-
      🎉 no goals
    -/


/-- the scalar multiplication of an homotopy -/
@[simps!]
def smul {R : Type*} [Semiring R] [Linear R V] (h : Homotopy f g) (a : R) :
    Homotopy (a • f) (a • g) where
  hom i j := a • h.hom i j
  zero i j hij := by
    /-
      ι : Type u_1
      V : Type u
      inst✝³ : CategoryTheory.Category.{v, u} V
      inst✝² : CategoryTheory.Preadditive V
      c : ComplexShape ι
      C D E : HomologicalComplex V c
      f g : Quiver.Hom C D
      h✝ k : Quiver.Hom D E
      i✝ : ι
      R : Type u_2
      inst✝¹ : Semiring R
      inst✝ : CategoryTheory.Linear R V
      h : Homotopy f g
      a : R
      i j : ι
      hij : Not (c.Rel j i)
      ⊢ Eq ((fun i j => HSMul.hSMul a (h.hom i j)) i j) 0
    -/
    dsimp
    /-
      ι : Type u_1
      V : Type u
      inst✝³ : CategoryTheory.Category.{v, u} V
      inst✝² : CategoryTheory.Preadditive V
      c : ComplexShape ι
      C D E : HomologicalComplex V c
      f g : Quiver.Hom C D
      h✝ k : Quiver.Hom D E
      i✝ : ι
      R : Type u_2
      inst✝¹ : Semiring R
      inst✝ : CategoryTheory.Linear R V
      h : Homotopy f g
      a : R
      i j : ι
      hij : Not (c.Rel j i)
      ⊢ Eq (HSMul.hSMul a (h.hom i j)) 0
    -/
    rw [h.zero i j hij, smul_zero]
    /-
      🎉 no goals
    -/
  comm i := by
    /-
      ι : Type u_1
      V : Type u
      inst✝³ : CategoryTheory.Category.{v, u} V
      inst✝² : CategoryTheory.Preadditive V
      c : ComplexShape ι
      C D E : HomologicalComplex V c
      f g : Quiver.Hom C D
      h✝ k : Quiver.Hom D E
      i✝ : ι
      R : Type u_2
      inst✝¹ : Semiring R
      inst✝ : CategoryTheory.Linear R V
      h : Homotopy f g
      a : R
      i : ι
      ⊢ Eq ((HSMul.hSMul a f).f i) (HAdd.hAdd (HAdd.hAdd ((dNext i) fun i j => HSMul …
    -/
    dsimp
    /-
      ι : Type u_1
      V : Type u
      inst✝³ : CategoryTheory.Category.{v, u} V
      inst✝² : CategoryTheory.Preadditive V
      c : ComplexShape ι
      C D E : HomologicalComplex V c
      f g : Quiver.Hom C D
      h✝ k : Quiver.Hom D E
      i✝ : ι
      R : Type u_2
      inst✝¹ : Semiring R
      inst✝ : CategoryTheory.Linear R V
      h : Homotopy f g
      a : R
      i : ι
      ⊢ Eq (HSMul.hSMul a (f.f i)) (HAdd.hAdd (HAdd.hAdd (CategoryTheory.CategoryStr …
    -/
    rw [h.comm]
    /-
      ι : Type u_1
      V : Type u
      inst✝³ : CategoryTheory.Category.{v, u} V
      inst✝² : CategoryTheory.Preadditive V
      c : ComplexShape ι
      C D E : HomologicalComplex V c
      f g : Quiver.Hom C D
      h✝ k : Quiver.Hom D E
      i✝ : ι
      R : Type u_2
      inst✝¹ : Semiring R
      inst✝ : CategoryTheory.Linear R V
      h : Homotopy f g
      a : R
      i : ι
      ⊢ Eq (HSMul.hSMul a (HAdd.hAdd (HAdd.hAdd ((dNext i) h.hom) ((prevD i) h.hom)) …
    -/
    dsimp [fromNext, toPrev]
    /-
      ι : Type u_1
      V : Type u
      inst✝³ : CategoryTheory.Category.{v, u} V
      inst✝² : CategoryTheory.Preadditive V
      c : ComplexShape ι
      C D E : HomologicalComplex V c
      f g : Quiver.Hom C D
      h✝ k : Quiver.Hom D E
      i✝ : ι
      R : Type u_2
      inst✝¹ : Semiring R
      inst✝ : CategoryTheory.Linear R V
      h : Homotopy f g
      a : R
      i : ι
      ⊢ Eq (HSMul.hSMul a (HAdd.hAdd (HAdd.hAdd (CategoryTheory.CategoryStruct.comp  …
    -/
    simp only [smul_add, Linear.comp_smul, Linear.smul_comp]
    /-
      🎉 no goals
    -/


/-- homotopy is closed under composition (on the right) -/
@[simps]
def compRight {e f : C ⟶ D} (h : Homotopy e f) (g : D ⟶ E) : Homotopy (e ≫ g) (f ≫ g) where
  hom i j := h.hom i j ≫ g.f j
                   /-
                     ι : Type u_1
                     V : Type u
                     inst✝¹ : CategoryTheory.Category.{v, u} V
                     inst✝ : CategoryTheory.Preadditive V
                     c : ComplexShape ι
                     C D E : HomologicalComplex V c
                     f✝ g✝ : Quiver.Hom C D
                     h✝ k : Quiver.Hom D E
                     i✝ : ι
                     e f : Quiver.Hom C D
                     h : Homotopy e f
                     g : Quiver.Hom D E
                     i j : ι
                     w : Not (c.Rel j i)
                     ⊢ Eq ((fun i j => CategoryTheory.CategoryStruct.comp (h.hom i j) (g.f j)) i j) 0
                   -/
  zero i j w := by dsimp; rw [h.zero i j w, zero_comp]
                          /-
                            🎉 no goals
                          -/
  comm i := by rw [comp_f, h.comm i, dNext_comp_right, prevD_comp_right, Preadditive.add_comp,
    comp_f, Preadditive.add_comp]


/-- homotopy is closed under composition (on the left) -/
@[simps]
def compLeft {f g : D ⟶ E} (h : Homotopy f g) (e : C ⟶ D) : Homotopy (e ≫ f) (e ≫ g) where
  hom i j := e.f i ≫ h.hom i j
                   /-
                     ι : Type u_1
                     V : Type u
                     inst✝¹ : CategoryTheory.Category.{v, u} V
                     inst✝ : CategoryTheory.Preadditive V
                     c : ComplexShape ι
                     C D E : HomologicalComplex V c
                     f✝ g✝ : Quiver.Hom C D
                     h✝ k : Quiver.Hom D E
                     i✝ : ι
                     f g : Quiver.Hom D E
                     h : Homotopy f g
                     e : Quiver.Hom C D
                     i j : ι
                     w : Not (c.Rel j i)
                     ⊢ Eq ((fun i j => CategoryTheory.CategoryStruct.comp (e.f i) (h.hom i j)) i j) 0
                   -/
  zero i j w := by dsimp; rw [h.zero i j w, comp_zero]
                          /-
                            🎉 no goals
                          -/
  comm i := by rw [comp_f, h.comm i, dNext_comp_left, prevD_comp_left, comp_f,
    Preadditive.comp_add, Preadditive.comp_add]


/-- homotopy is closed under composition -/
@[simps!]
def comp {C₁ C₂ C₃ : HomologicalComplex V c} {f₁ g₁ : C₁ ⟶ C₂} {f₂ g₂ : C₂ ⟶ C₃}
    (h₁ : Homotopy f₁ g₁) (h₂ : Homotopy f₂ g₂) : Homotopy (f₁ ≫ f₂) (g₁ ≫ g₂) :=
  (h₁.compRight _).trans (h₂.compLeft _)


/-- a variant of `Homotopy.compRight` useful for dealing with homotopy equivalences. -/
@[simps!]
def compRightId {f : C ⟶ C} (h : Homotopy f (𝟙 C)) (g : C ⟶ D) : Homotopy (f ≫ g) g :=
  (h.compRight g).trans (ofEq <| id_comp _)


/-- a variant of `Homotopy.compLeft` useful for dealing with homotopy equivalences. -/
@[simps!]
def compLeftId {f : D ⟶ D} (h : Homotopy f (𝟙 D)) (g : C ⟶ D) : Homotopy (g ≫ f) g :=
  (h.compLeft g).trans (ofEq <| comp_id _)


/-- The null homotopic map associated to a family `hom` of morphisms `C_i ⟶ D_j`.
This is the same datum as for the field `hom` in the structure `Homotopy`. For
this definition, we do not need the field `zero` of that structure
as this definition uses only the maps `C_i ⟶ C_j` when `c.Rel j i`. -/
def nullHomotopicMap (hom : ∀ i j, C.X i ⟶ D.X j) : C ⟶ D where
  f i := dNext i hom + prevD i hom
  comm' i j hij := by
    have eq1 : prevD i hom ≫ D.d i j = 0 := by
      simp only [prevD, AddMonoidHom.mk'_apply, assoc, d_comp_d, comp_zero]
    have eq2 : C.d i j ≫ dNext j hom = 0 := by
      simp only [dNext, AddMonoidHom.mk'_apply, d_comp_d_assoc, zero_comp]
    /-
      ι : Type u_1
      V : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} V
      inst✝ : CategoryTheory.Preadditive V
      c : ComplexShape ι
      C D E : HomologicalComplex V c
      f g : Quiver.Hom C D
      h k : Quiver.Hom D E
      i✝ : ι
      hom : (i j : ι) → Quiver.Hom (C.X i) (D.X j)
      i j : ι
      hij : c.Rel i j
      eq1 : Eq (CategoryTheory.CategoryStruct.comp ((prevD i) hom) (D.d i j)) 0
      eq2 : Eq (CategoryTheory.CategoryStruct.comp (C.d i j) ((dNext j) hom)) 0
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun i => HAdd.hAdd ((dNext i) hom)  …
    -/
    dsimp only
    rw [dNext_eq hom hij, prevD_eq hom hij, Preadditive.comp_add, Preadditive.add_comp, eq1, eq2,
      add_zero, zero_add, assoc]


open Classical in
/-- Variant of `nullHomotopicMap` where the input consists only of the
relevant maps `C_i ⟶ D_j` such that `c.Rel j i`. -/
def nullHomotopicMap' (h : ∀ i j, c.Rel j i → (C.X i ⟶ D.X j)) : C ⟶ D :=
  nullHomotopicMap fun i j => dite (c.Rel j i) (h i j) fun _ => 0


/-- Compatibility of `nullHomotopicMap` with the postcomposition by a morphism
of complexes. -/
theorem nullHomotopicMap_comp (hom : ∀ i j, C.X i ⟶ D.X j) (g : D ⟶ E) :
    nullHomotopicMap hom ≫ g = nullHomotopicMap fun i j => hom i j ≫ g.f j := by
  /-
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    c : ComplexShape ι
    C D E : HomologicalComplex V c
    hom : (i j : ι) → Quiver.Hom (C.X i) (D.X j)
    g : Quiver.Hom D E
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (Homotopy.nullHomotopicMap hom) g) (H …
  -/
  ext n
  /-
    case h
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    c : ComplexShape ι
    C D E : HomologicalComplex V c
    hom : (i j : ι) → Quiver.Hom (C.X i) (D.X j)
    g : Quiver.Hom D E
    n : ι
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp (Homotopy.nullHomotopicMap hom) g).f …
  -/
  dsimp [nullHomotopicMap, fromNext, toPrev, AddMonoidHom.mk'_apply]
  /-
    case h
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    c : ComplexShape ι
    C D E : HomologicalComplex V c
    hom : (i j : ι) → Quiver.Hom (C.X i) (D.X j)
    g : Quiver.Hom D E
    n : ι
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HAdd.hAdd (CategoryTheory.CategorySt …
  -/
  simp only [Preadditive.add_comp, assoc, g.comm]
  /-
    🎉 no goals
  -/


/-- Compatibility of `nullHomotopicMap'` with the postcomposition by a morphism
of complexes. -/
theorem nullHomotopicMap'_comp (hom : ∀ i j, c.Rel j i → (C.X i ⟶ D.X j)) (g : D ⟶ E) :
    nullHomotopicMap' hom ≫ g = nullHomotopicMap' fun i j hij => hom i j hij ≫ g.f j := by
  /-
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    c : ComplexShape ι
    C D E : HomologicalComplex V c
    hom : (i j : ι) → c.Rel j i → Quiver.Hom (C.X i) (D.X j)
    g : Quiver.Hom D E
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (Homotopy.nullHomotopicMap' hom) g) ( …
  -/
  ext n
  /-
    case h
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    c : ComplexShape ι
    C D E : HomologicalComplex V c
    hom : (i j : ι) → c.Rel j i → Quiver.Hom (C.X i) (D.X j)
    g : Quiver.Hom D E
    n : ι
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp (Homotopy.nullHomotopicMap' hom) g). …
  -/
  erw [nullHomotopicMap_comp]
  /-
    case h
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    c : ComplexShape ι
    C D E : HomologicalComplex V c
    hom : (i j : ι) → c.Rel j i → Quiver.Hom (C.X i) (D.X j)
    g : Quiver.Hom D E
    n : ι
    ⊢ Eq ((Homotopy.nullHomotopicMap fun i j => CategoryTheory.CategoryStruct.comp …
  -/
  congr
  /-
    case h.e_self.e_hom
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    c : ComplexShape ι
    C D E : HomologicalComplex V c
    hom : (i j : ι) → c.Rel j i → Quiver.Hom (C.X i) (D.X j)
    g : Quiver.Hom D E
    n : ι
    ⊢ Eq (fun i j => CategoryTheory.CategoryStruct.comp (dite (c.Rel j i) (hom i j …
  -/
  ext i j
  /-
    case h.e_self.e_hom.h.h
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    c : ComplexShape ι
    C D E : HomologicalComplex V c
    hom : (i j : ι) → c.Rel j i → Quiver.Hom (C.X i) (D.X j)
    g : Quiver.Hom D E
    n i j : ι
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (dite (c.Rel j i) (hom i j) fun x =>  …
  -/
  split_ifs
    /-
      case pos
      ι : Type u_1
      V : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} V
      inst✝ : CategoryTheory.Preadditive V
      c : ComplexShape ι
      C D E : HomologicalComplex V c
      hom : (i j : ι) → c.Rel j i → Quiver.Hom (C.X i) (D.X j)
      g : Quiver.Hom D E
      n i j : ι
      h✝ : c.Rel j i
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (hom i j h✝) (g.f j)) ((fun i j hij = …
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case neg
      ι : Type u_1
      V : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} V
      inst✝ : CategoryTheory.Preadditive V
      c : ComplexShape ι
      C D E : HomologicalComplex V c
      hom : (i j : ι) → c.Rel j i → Quiver.Hom (C.X i) (D.X j)
      g : Quiver.Hom D E
      n i j : ι
      h✝ : Not (c.Rel j i)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp 0 (g.f j)) 0
    -/
  · rw [zero_comp]
    /-
      🎉 no goals
    -/


/-- Compatibility of `nullHomotopicMap` with the precomposition by a morphism
of complexes. -/
theorem comp_nullHomotopicMap (f : C ⟶ D) (hom : ∀ i j, D.X i ⟶ E.X j) :
    f ≫ nullHomotopicMap hom = nullHomotopicMap fun i j => f.f i ≫ hom i j := by
  /-
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    c : ComplexShape ι
    C D E : HomologicalComplex V c
    f : Quiver.Hom C D
    hom : (i j : ι) → Quiver.Hom (D.X i) (E.X j)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp f (Homotopy.nullHomotopicMap hom)) (H …
  -/
  ext n
  /-
    case h
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    c : ComplexShape ι
    C D E : HomologicalComplex V c
    f : Quiver.Hom C D
    hom : (i j : ι) → Quiver.Hom (D.X i) (E.X j)
    n : ι
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp f (Homotopy.nullHomotopicMap hom)).f …
  -/
  dsimp [nullHomotopicMap, fromNext, toPrev, AddMonoidHom.mk'_apply]
  /-
    case h
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    c : ComplexShape ι
    C D E : HomologicalComplex V c
    f : Quiver.Hom C D
    hom : (i j : ι) → Quiver.Hom (D.X i) (E.X j)
    n : ι
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (f.f n) (HAdd.hAdd (CategoryTheory.Ca …
  -/
  simp only [Preadditive.comp_add, assoc, f.comm_assoc]
  /-
    🎉 no goals
  -/


/-- Compatibility of `nullHomotopicMap'` with the precomposition by a morphism
of complexes. -/
theorem comp_nullHomotopicMap' (f : C ⟶ D) (hom : ∀ i j, c.Rel j i → (D.X i ⟶ E.X j)) :
    f ≫ nullHomotopicMap' hom = nullHomotopicMap' fun i j hij => f.f i ≫ hom i j hij := by
  /-
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    c : ComplexShape ι
    C D E : HomologicalComplex V c
    f : Quiver.Hom C D
    hom : (i j : ι) → c.Rel j i → Quiver.Hom (D.X i) (E.X j)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp f (Homotopy.nullHomotopicMap' hom)) ( …
  -/
  ext n
  /-
    case h
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    c : ComplexShape ι
    C D E : HomologicalComplex V c
    f : Quiver.Hom C D
    hom : (i j : ι) → c.Rel j i → Quiver.Hom (D.X i) (E.X j)
    n : ι
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp f (Homotopy.nullHomotopicMap' hom)). …
  -/
  erw [comp_nullHomotopicMap]
  /-
    case h
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    c : ComplexShape ι
    C D E : HomologicalComplex V c
    f : Quiver.Hom C D
    hom : (i j : ι) → c.Rel j i → Quiver.Hom (D.X i) (E.X j)
    n : ι
    ⊢ Eq ((Homotopy.nullHomotopicMap fun i j => CategoryTheory.CategoryStruct.comp …
  -/
  congr
  /-
    case h.e_self.e_hom
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    c : ComplexShape ι
    C D E : HomologicalComplex V c
    f : Quiver.Hom C D
    hom : (i j : ι) → c.Rel j i → Quiver.Hom (D.X i) (E.X j)
    n : ι
    ⊢ Eq (fun i j => CategoryTheory.CategoryStruct.comp (f.f i) (dite (c.Rel j i)  …
  -/
  ext i j
  /-
    case h.e_self.e_hom.h.h
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    c : ComplexShape ι
    C D E : HomologicalComplex V c
    f : Quiver.Hom C D
    hom : (i j : ι) → c.Rel j i → Quiver.Hom (D.X i) (E.X j)
    n i j : ι
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (f.f i) (dite (c.Rel j i) (hom i j) f …
  -/
  split_ifs
    /-
      case pos
      ι : Type u_1
      V : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} V
      inst✝ : CategoryTheory.Preadditive V
      c : ComplexShape ι
      C D E : HomologicalComplex V c
      f : Quiver.Hom C D
      hom : (i j : ι) → c.Rel j i → Quiver.Hom (D.X i) (E.X j)
      n i j : ι
      h✝ : c.Rel j i
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (f.f i) (hom i j h✝)) ((fun i j hij = …
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case neg
      ι : Type u_1
      V : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} V
      inst✝ : CategoryTheory.Preadditive V
      c : ComplexShape ι
      C D E : HomologicalComplex V c
      f : Quiver.Hom C D
      hom : (i j : ι) → c.Rel j i → Quiver.Hom (D.X i) (E.X j)
      n i j : ι
      h✝ : Not (c.Rel j i)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (f.f i) 0) 0
    -/
  · rw [comp_zero]
    /-
      🎉 no goals
    -/


/-- Compatibility of `nullHomotopicMap` with the application of additive functors -/
theorem map_nullHomotopicMap {W : Type*} [Category W] [Preadditive W] (G : V ⥤ W) [G.Additive]
    (hom : ∀ i j, C.X i ⟶ D.X j) :
    (G.mapHomologicalComplex c).map (nullHomotopicMap hom) =
                                      /-
                                        ι : Type u_1
                                        V : Type u
                                        inst✝⁴ : CategoryTheory.Category.{v, u} V
                                        inst✝³ : CategoryTheory.Preadditive V
                                        c : ComplexShape ι
                                        C D E : HomologicalComplex V c
                                        f g : Quiver.Hom C D
                                        h k : Quiver.Hom D E
                                        i✝ : ι
                                        W : Type u_2
                                        inst✝² : CategoryTheory.Category.{?u.118229, u_2} W
                                        inst✝¹ : CategoryTheory.Preadditive W
                                        G : CategoryTheory.Functor V W
                                        inst✝ : G.Additive
                                        hom : (i j : ι) → Quiver.Hom (C.X i) (D.X j)
                                        i j : ι
                                        ⊢ Quiver.Hom (((G.mapHomologicalComplex c).obj C).X i) (((G.mapHomologicalComp …
                                      -/
      nullHomotopicMap (fun i j => by exact G.map (hom i j)) := by
                                      /-
                                        🎉 no goals
                                      -/
  /-
    ι : Type u_1
    V : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} V
    inst✝³ : CategoryTheory.Preadditive V
    c : ComplexShape ι
    C D : HomologicalComplex V c
    W : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_2} W
    inst✝¹ : CategoryTheory.Preadditive W
    G : CategoryTheory.Functor V W
    inst✝ : G.Additive
    hom : (i j : ι) → Quiver.Hom (C.X i) (D.X j)
    ⊢ Eq ((G.mapHomologicalComplex c).map (Homotopy.nullHomotopicMap hom)) (Homoto …
  -/
  ext i
  /-
    case h
    ι : Type u_1
    V : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} V
    inst✝³ : CategoryTheory.Preadditive V
    c : ComplexShape ι
    C D : HomologicalComplex V c
    W : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_2} W
    inst✝¹ : CategoryTheory.Preadditive W
    G : CategoryTheory.Functor V W
    inst✝ : G.Additive
    hom : (i j : ι) → Quiver.Hom (C.X i) (D.X j)
    i : ι
    ⊢ Eq (((G.mapHomologicalComplex c).map (Homotopy.nullHomotopicMap hom)).f i) ( …
  -/
  dsimp [nullHomotopicMap, dNext, prevD]
  /-
    case h
    ι : Type u_1
    V : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} V
    inst✝³ : CategoryTheory.Preadditive V
    c : ComplexShape ι
    C D : HomologicalComplex V c
    W : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_2} W
    inst✝¹ : CategoryTheory.Preadditive W
    G : CategoryTheory.Functor V W
    inst✝ : G.Additive
    hom : (i j : ι) → Quiver.Hom (C.X i) (D.X j)
    i : ι
    ⊢ Eq (G.map (HAdd.hAdd (CategoryTheory.CategoryStruct.comp (C.d i (c.next i))  …
  -/
  simp only [G.map_comp, Functor.map_add]
  /-
    🎉 no goals
  -/


/-- Compatibility of `nullHomotopicMap'` with the application of additive functors -/
theorem map_nullHomotopicMap' {W : Type*} [Category W] [Preadditive W] (G : V ⥤ W) [G.Additive]
    (hom : ∀ i j, c.Rel j i → (C.X i ⟶ D.X j)) :
    (G.mapHomologicalComplex c).map (nullHomotopicMap' hom) =
                                          /-
                                            ι : Type u_1
                                            V : Type u
                                            inst✝⁴ : CategoryTheory.Category.{v, u} V
                                            inst✝³ : CategoryTheory.Preadditive V
                                            c : ComplexShape ι
                                            C D E : HomologicalComplex V c
                                            f g : Quiver.Hom C D
                                            h k : Quiver.Hom D E
                                            i✝ : ι
                                            W : Type u_2
                                            inst✝² : CategoryTheory.Category.{?u.121611, u_2} W
                                            inst✝¹ : CategoryTheory.Preadditive W
                                            G : CategoryTheory.Functor V W
                                            inst✝ : G.Additive
                                            hom : (i j : ι) → c.Rel j i → Quiver.Hom (C.X i) (D.X j)
                                            i j : ι
                                            hij : c.Rel j i
                                            ⊢ Quiver.Hom (((G.mapHomologicalComplex c).obj C).X i) (((G.mapHomologicalComp …
                                          -/
      nullHomotopicMap' fun i j hij => by exact G.map (hom i j hij) := by
                                          /-
                                            🎉 no goals
                                          -/
  /-
    ι : Type u_1
    V : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} V
    inst✝³ : CategoryTheory.Preadditive V
    c : ComplexShape ι
    C D : HomologicalComplex V c
    W : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_2} W
    inst✝¹ : CategoryTheory.Preadditive W
    G : CategoryTheory.Functor V W
    inst✝ : G.Additive
    hom : (i j : ι) → c.Rel j i → Quiver.Hom (C.X i) (D.X j)
    ⊢ Eq ((G.mapHomologicalComplex c).map (Homotopy.nullHomotopicMap' hom)) (Homot …
  -/
  ext n
  /-
    case h
    ι : Type u_1
    V : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} V
    inst✝³ : CategoryTheory.Preadditive V
    c : ComplexShape ι
    C D : HomologicalComplex V c
    W : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_2} W
    inst✝¹ : CategoryTheory.Preadditive W
    G : CategoryTheory.Functor V W
    inst✝ : G.Additive
    hom : (i j : ι) → c.Rel j i → Quiver.Hom (C.X i) (D.X j)
    n : ι
    ⊢ Eq (((G.mapHomologicalComplex c).map (Homotopy.nullHomotopicMap' hom)).f n)  …
  -/
  erw [map_nullHomotopicMap]
  /-
    case h
    ι : Type u_1
    V : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} V
    inst✝³ : CategoryTheory.Preadditive V
    c : ComplexShape ι
    C D : HomologicalComplex V c
    W : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_2} W
    inst✝¹ : CategoryTheory.Preadditive W
    G : CategoryTheory.Functor V W
    inst✝ : G.Additive
    hom : (i j : ι) → c.Rel j i → Quiver.Hom (C.X i) (D.X j)
    n : ι
    ⊢ Eq ((Homotopy.nullHomotopicMap fun i j => G.map (dite (c.Rel j i) (hom i j)  …
  -/
  congr
  /-
    case h.e_self.e_hom
    ι : Type u_1
    V : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} V
    inst✝³ : CategoryTheory.Preadditive V
    c : ComplexShape ι
    C D : HomologicalComplex V c
    W : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_2} W
    inst✝¹ : CategoryTheory.Preadditive W
    G : CategoryTheory.Functor V W
    inst✝ : G.Additive
    hom : (i j : ι) → c.Rel j i → Quiver.Hom (C.X i) (D.X j)
    n : ι
    ⊢ Eq (fun i j => G.map (dite (c.Rel j i) (hom i j) fun x => 0)) fun i j => dit …
  -/
  ext i j
  /-
    case h.e_self.e_hom.h.h
    ι : Type u_1
    V : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} V
    inst✝³ : CategoryTheory.Preadditive V
    c : ComplexShape ι
    C D : HomologicalComplex V c
    W : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_2} W
    inst✝¹ : CategoryTheory.Preadditive W
    G : CategoryTheory.Functor V W
    inst✝ : G.Additive
    hom : (i j : ι) → c.Rel j i → Quiver.Hom (C.X i) (D.X j)
    n i j : ι
    ⊢ Eq (G.map (dite (c.Rel j i) (hom i j) fun x => 0)) (dite (c.Rel j i) ((fun i …
  -/
  split_ifs
    /-
      case pos
      ι : Type u_1
      V : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} V
      inst✝³ : CategoryTheory.Preadditive V
      c : ComplexShape ι
      C D : HomologicalComplex V c
      W : Type u_2
      inst✝² : CategoryTheory.Category.{u_3, u_2} W
      inst✝¹ : CategoryTheory.Preadditive W
      G : CategoryTheory.Functor V W
      inst✝ : G.Additive
      hom : (i j : ι) → c.Rel j i → Quiver.Hom (C.X i) (D.X j)
      n i j : ι
      h✝ : c.Rel j i
      ⊢ Eq (G.map (hom i j h✝)) ((fun i j hij => G.map (hom i j hij)) i j h✝)
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case neg
      ι : Type u_1
      V : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} V
      inst✝³ : CategoryTheory.Preadditive V
      c : ComplexShape ι
      C D : HomologicalComplex V c
      W : Type u_2
      inst✝² : CategoryTheory.Category.{u_3, u_2} W
      inst✝¹ : CategoryTheory.Preadditive W
      G : CategoryTheory.Functor V W
      inst✝ : G.Additive
      hom : (i j : ι) → c.Rel j i → Quiver.Hom (C.X i) (D.X j)
      n i j : ι
      h✝ : Not (c.Rel j i)
      ⊢ Eq (G.map 0) 0
    -/
  · rw [G.map_zero]
    /-
      🎉 no goals
    -/


/-- Tautological construction of the `Homotopy` to zero for maps constructed by
`nullHomotopicMap`, at least when we have the `zero` condition. -/
@[simps]
def nullHomotopy (hom : ∀ i j, C.X i ⟶ D.X j) (zero : ∀ i j, ¬c.Rel j i → hom i j = 0) :
    Homotopy (nullHomotopicMap hom) 0 :=
  { hom := hom
    zero := zero
    comm := by
      /-
        ι : Type u_1
        V : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} V
        inst✝ : CategoryTheory.Preadditive V
        c : ComplexShape ι
        C D E : HomologicalComplex V c
        f g : Quiver.Hom C D
        h k : Quiver.Hom D E
        i : ι
        hom : (i j : ι) → Quiver.Hom (C.X i) (D.X j)
        zero : ∀ (i j : ι), Not (c.Rel j i) → Eq (hom i j) 0
        ⊢ ∀ (i : ι), Eq ((Homotopy.nullHomotopicMap hom).f i) (HAdd.hAdd (HAdd.hAdd (( …
      -/
      intro i
      /-
        ι : Type u_1
        V : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} V
        inst✝ : CategoryTheory.Preadditive V
        c : ComplexShape ι
        C D E : HomologicalComplex V c
        f g : Quiver.Hom C D
        h k : Quiver.Hom D E
        i✝ : ι
        hom : (i j : ι) → Quiver.Hom (C.X i) (D.X j)
        zero : ∀ (i j : ι), Not (c.Rel j i) → Eq (hom i j) 0
        i : ι
        ⊢ Eq ((Homotopy.nullHomotopicMap hom).f i) (HAdd.hAdd (HAdd.hAdd ((dNext i) ho …
      -/
      rw [HomologicalComplex.zero_f_apply, add_zero]
      /-
        ι : Type u_1
        V : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} V
        inst✝ : CategoryTheory.Preadditive V
        c : ComplexShape ι
        C D E : HomologicalComplex V c
        f g : Quiver.Hom C D
        h k : Quiver.Hom D E
        i✝ : ι
        hom : (i j : ι) → Quiver.Hom (C.X i) (D.X j)
        zero : ∀ (i j : ι), Not (c.Rel j i) → Eq (hom i j) 0
        i : ι
        ⊢ Eq ((Homotopy.nullHomotopicMap hom).f i) (HAdd.hAdd ((dNext i) hom) ((prevD  …
      -/
      rfl }
      /-
        🎉 no goals
      -/


open Classical in
/-- Homotopy to zero for maps constructed with `nullHomotopicMap'` -/
@[simps!]
def nullHomotopy' (h : ∀ i j, c.Rel j i → (C.X i ⟶ D.X j)) : Homotopy (nullHomotopicMap' h) 0 := by
  /-
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    c : ComplexShape ι
    C D E : HomologicalComplex V c
    f g : Quiver.Hom C D
    h✝ k : Quiver.Hom D E
    i : ι
    h : (i j : ι) → c.Rel j i → Quiver.Hom (C.X i) (D.X j)
    ⊢ Homotopy (Homotopy.nullHomotopicMap' h) 0
  -/
  apply nullHomotopy fun i j => dite (c.Rel j i) (h i j) fun _ => 0
  /-
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    c : ComplexShape ι
    C D E : HomologicalComplex V c
    f g : Quiver.Hom C D
    h✝ k : Quiver.Hom D E
    i : ι
    h : (i j : ι) → c.Rel j i → Quiver.Hom (C.X i) (D.X j)
    ⊢ ∀ (i j : ι), Not (c.Rel j i) → Eq (dite (c.Rel j i) (h i j) fun x => 0) 0
  -/
  intro i j hij
  /-
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    c : ComplexShape ι
    C D E : HomologicalComplex V c
    f g : Quiver.Hom C D
    h✝ k : Quiver.Hom D E
    i✝ : ι
    h : (i j : ι) → c.Rel j i → Quiver.Hom (C.X i) (D.X j)
    i j : ι
    hij : Not (c.Rel j i)
    ⊢ Eq (dite (c.Rel j i) (h i j) fun x => 0) 0
  -/
  rw [dite_eq_right_iff]
  /-
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    c : ComplexShape ι
    C D E : HomologicalComplex V c
    f g : Quiver.Hom C D
    h✝ k : Quiver.Hom D E
    i✝ : ι
    h : (i j : ι) → c.Rel j i → Quiver.Hom (C.X i) (D.X j)
    i j : ι
    hij : Not (c.Rel j i)
    ⊢ ∀ (h_1 : c.Rel j i), Eq (h i j h_1) 0
  -/
  intro hij'
  /-
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    c : ComplexShape ι
    C D E : HomologicalComplex V c
    f g : Quiver.Hom C D
    h✝ k : Quiver.Hom D E
    i✝ : ι
    h : (i j : ι) → c.Rel j i → Quiver.Hom (C.X i) (D.X j)
    i j : ι
    hij : Not (c.Rel j i)
    hij' : c.Rel j i
    ⊢ Eq (h i j hij') 0
  -/
  exfalso
  /-
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    c : ComplexShape ι
    C D E : HomologicalComplex V c
    f g : Quiver.Hom C D
    h✝ k : Quiver.Hom D E
    i✝ : ι
    h : (i j : ι) → c.Rel j i → Quiver.Hom (C.X i) (D.X j)
    i j : ι
    hij : Not (c.Rel j i)
    hij' : c.Rel j i
    ⊢ False
  -/
  exact hij hij'
  /-
    🎉 no goals
  -/


@[simp]
theorem nullHomotopicMap_f {k₂ k₁ k₀ : ι} (r₂₁ : c.Rel k₂ k₁) (r₁₀ : c.Rel k₁ k₀)
    (hom : ∀ i j, C.X i ⟶ D.X j) :
    (nullHomotopicMap hom).f k₁ = C.d k₁ k₀ ≫ hom k₀ k₁ + hom k₁ k₂ ≫ D.d k₂ k₁ := by
  /-
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    c : ComplexShape ι
    C D : HomologicalComplex V c
    k₂ k₁ k₀ : ι
    r₂₁ : c.Rel k₂ k₁
    r₁₀ : c.Rel k₁ k₀
    hom : (i j : ι) → Quiver.Hom (C.X i) (D.X j)
    ⊢ Eq ((Homotopy.nullHomotopicMap hom).f k₁) (HAdd.hAdd (CategoryTheory.Categor …
  -/
  dsimp only [nullHomotopicMap]
  /-
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    c : ComplexShape ι
    C D : HomologicalComplex V c
    k₂ k₁ k₀ : ι
    r₂₁ : c.Rel k₂ k₁
    r₁₀ : c.Rel k₁ k₀
    hom : (i j : ι) → Quiver.Hom (C.X i) (D.X j)
    ⊢ Eq (HAdd.hAdd ((dNext k₁) hom) ((prevD k₁) hom)) (HAdd.hAdd (CategoryTheory. …
  -/
  rw [dNext_eq hom r₁₀, prevD_eq hom r₂₁]
  /-
    🎉 no goals
  -/


@[simp]
theorem nullHomotopicMap'_f {k₂ k₁ k₀ : ι} (r₂₁ : c.Rel k₂ k₁) (r₁₀ : c.Rel k₁ k₀)
    (h : ∀ i j, c.Rel j i → (C.X i ⟶ D.X j)) :
    (nullHomotopicMap' h).f k₁ = C.d k₁ k₀ ≫ h k₀ k₁ r₁₀ + h k₁ k₂ r₂₁ ≫ D.d k₂ k₁ := by
  /-
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    c : ComplexShape ι
    C D : HomologicalComplex V c
    k₂ k₁ k₀ : ι
    r₂₁ : c.Rel k₂ k₁
    r₁₀ : c.Rel k₁ k₀
    h : (i j : ι) → c.Rel j i → Quiver.Hom (C.X i) (D.X j)
    ⊢ Eq ((Homotopy.nullHomotopicMap' h).f k₁) (HAdd.hAdd (CategoryTheory.Category …
  -/
  simp only [nullHomotopicMap']
  /-
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    c : ComplexShape ι
    C D : HomologicalComplex V c
    k₂ k₁ k₀ : ι
    r₂₁ : c.Rel k₂ k₁
    r₁₀ : c.Rel k₁ k₀
    h : (i j : ι) → c.Rel j i → Quiver.Hom (C.X i) (D.X j)
    ⊢ Eq ((Homotopy.nullHomotopicMap fun i j => dite (c.Rel j i) (h i j) fun x =>  …
  -/
  rw [nullHomotopicMap_f r₂₁ r₁₀]
  /-
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    c : ComplexShape ι
    C D : HomologicalComplex V c
    k₂ k₁ k₀ : ι
    r₂₁ : c.Rel k₂ k₁
    r₁₀ : c.Rel k₁ k₀
    h : (i j : ι) → c.Rel j i → Quiver.Hom (C.X i) (D.X j)
    ⊢ Eq (HAdd.hAdd (CategoryTheory.CategoryStruct.comp (C.d k₁ k₀) (dite (c.Rel k …
  -/
  split_ifs
  /-
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    c : ComplexShape ι
    C D : HomologicalComplex V c
    k₂ k₁ k₀ : ι
    r₂₁ : c.Rel k₂ k₁
    r₁₀ : c.Rel k₁ k₀
    h : (i j : ι) → c.Rel j i → Quiver.Hom (C.X i) (D.X j)
    ⊢ Eq (HAdd.hAdd (CategoryTheory.CategoryStruct.comp (C.d k₁ k₀) (h k₀ k₁ r₁₀)) …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem nullHomotopicMap_f_of_not_rel_left {k₁ k₀ : ι} (r₁₀ : c.Rel k₁ k₀)
    (hk₀ : ∀ l : ι, ¬c.Rel k₀ l) (hom : ∀ i j, C.X i ⟶ D.X j) :
    (nullHomotopicMap hom).f k₀ = hom k₀ k₁ ≫ D.d k₁ k₀ := by
  /-
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    c : ComplexShape ι
    C D : HomologicalComplex V c
    k₁ k₀ : ι
    r₁₀ : c.Rel k₁ k₀
    hk₀ : ∀ (l : ι), Not (c.Rel k₀ l)
    hom : (i j : ι) → Quiver.Hom (C.X i) (D.X j)
    ⊢ Eq ((Homotopy.nullHomotopicMap hom).f k₀) (CategoryTheory.CategoryStruct.com …
  -/
  dsimp only [nullHomotopicMap]
  /-
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    c : ComplexShape ι
    C D : HomologicalComplex V c
    k₁ k₀ : ι
    r₁₀ : c.Rel k₁ k₀
    hk₀ : ∀ (l : ι), Not (c.Rel k₀ l)
    hom : (i j : ι) → Quiver.Hom (C.X i) (D.X j)
    ⊢ Eq (HAdd.hAdd ((dNext k₀) hom) ((prevD k₀) hom)) (CategoryTheory.CategoryStr …
  -/
  rw [prevD_eq hom r₁₀, dNext, AddMonoidHom.mk'_apply, C.shape, zero_comp, zero_add]
  /-
    case a
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    c : ComplexShape ι
    C D : HomologicalComplex V c
    k₁ k₀ : ι
    r₁₀ : c.Rel k₁ k₀
    hk₀ : ∀ (l : ι), Not (c.Rel k₀ l)
    hom : (i j : ι) → Quiver.Hom (C.X i) (D.X j)
    ⊢ Not (c.Rel k₀ (c.next k₀))
  -/
  exact hk₀ _
  /-
    🎉 no goals
  -/


@[simp]
theorem nullHomotopicMap'_f_of_not_rel_left {k₁ k₀ : ι} (r₁₀ : c.Rel k₁ k₀)
    (hk₀ : ∀ l : ι, ¬c.Rel k₀ l) (h : ∀ i j, c.Rel j i → (C.X i ⟶ D.X j)) :
    (nullHomotopicMap' h).f k₀ = h k₀ k₁ r₁₀ ≫ D.d k₁ k₀ := by
  /-
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    c : ComplexShape ι
    C D : HomologicalComplex V c
    k₁ k₀ : ι
    r₁₀ : c.Rel k₁ k₀
    hk₀ : ∀ (l : ι), Not (c.Rel k₀ l)
    h : (i j : ι) → c.Rel j i → Quiver.Hom (C.X i) (D.X j)
    ⊢ Eq ((Homotopy.nullHomotopicMap' h).f k₀) (CategoryTheory.CategoryStruct.comp …
  -/
  simp only [nullHomotopicMap']
  /-
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    c : ComplexShape ι
    C D : HomologicalComplex V c
    k₁ k₀ : ι
    r₁₀ : c.Rel k₁ k₀
    hk₀ : ∀ (l : ι), Not (c.Rel k₀ l)
    h : (i j : ι) → c.Rel j i → Quiver.Hom (C.X i) (D.X j)
    ⊢ Eq ((Homotopy.nullHomotopicMap fun i j => dite (c.Rel j i) (h i j) fun x =>  …
  -/
  rw [nullHomotopicMap_f_of_not_rel_left r₁₀ hk₀]
  /-
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    c : ComplexShape ι
    C D : HomologicalComplex V c
    k₁ k₀ : ι
    r₁₀ : c.Rel k₁ k₀
    hk₀ : ∀ (l : ι), Not (c.Rel k₀ l)
    h : (i j : ι) → c.Rel j i → Quiver.Hom (C.X i) (D.X j)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (dite (c.Rel k₁ k₀) (h k₀ k₁) fun x = …
  -/
  split_ifs
  /-
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    c : ComplexShape ι
    C D : HomologicalComplex V c
    k₁ k₀ : ι
    r₁₀ : c.Rel k₁ k₀
    hk₀ : ∀ (l : ι), Not (c.Rel k₀ l)
    h : (i j : ι) → c.Rel j i → Quiver.Hom (C.X i) (D.X j)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (h k₀ k₁ r₁₀) (D.d k₁ k₀)) (CategoryT …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem nullHomotopicMap_f_of_not_rel_right {k₁ k₀ : ι} (r₁₀ : c.Rel k₁ k₀)
    (hk₁ : ∀ l : ι, ¬c.Rel l k₁) (hom : ∀ i j, C.X i ⟶ D.X j) :
    (nullHomotopicMap hom).f k₁ = C.d k₁ k₀ ≫ hom k₀ k₁ := by
  /-
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    c : ComplexShape ι
    C D : HomologicalComplex V c
    k₁ k₀ : ι
    r₁₀ : c.Rel k₁ k₀
    hk₁ : ∀ (l : ι), Not (c.Rel l k₁)
    hom : (i j : ι) → Quiver.Hom (C.X i) (D.X j)
    ⊢ Eq ((Homotopy.nullHomotopicMap hom).f k₁) (CategoryTheory.CategoryStruct.com …
  -/
  dsimp only [nullHomotopicMap]
  /-
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    c : ComplexShape ι
    C D : HomologicalComplex V c
    k₁ k₀ : ι
    r₁₀ : c.Rel k₁ k₀
    hk₁ : ∀ (l : ι), Not (c.Rel l k₁)
    hom : (i j : ι) → Quiver.Hom (C.X i) (D.X j)
    ⊢ Eq (HAdd.hAdd ((dNext k₁) hom) ((prevD k₁) hom)) (CategoryTheory.CategoryStr …
  -/
  rw [dNext_eq hom r₁₀, prevD, AddMonoidHom.mk'_apply, D.shape, comp_zero, add_zero]
  /-
    case a
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    c : ComplexShape ι
    C D : HomologicalComplex V c
    k₁ k₀ : ι
    r₁₀ : c.Rel k₁ k₀
    hk₁ : ∀ (l : ι), Not (c.Rel l k₁)
    hom : (i j : ι) → Quiver.Hom (C.X i) (D.X j)
    ⊢ Not (c.Rel (c.prev k₁) k₁)
  -/
  exact hk₁ _
  /-
    🎉 no goals
  -/


@[simp]
theorem nullHomotopicMap'_f_of_not_rel_right {k₁ k₀ : ι} (r₁₀ : c.Rel k₁ k₀)
    (hk₁ : ∀ l : ι, ¬c.Rel l k₁) (h : ∀ i j, c.Rel j i → (C.X i ⟶ D.X j)) :
    (nullHomotopicMap' h).f k₁ = C.d k₁ k₀ ≫ h k₀ k₁ r₁₀ := by
  /-
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    c : ComplexShape ι
    C D : HomologicalComplex V c
    k₁ k₀ : ι
    r₁₀ : c.Rel k₁ k₀
    hk₁ : ∀ (l : ι), Not (c.Rel l k₁)
    h : (i j : ι) → c.Rel j i → Quiver.Hom (C.X i) (D.X j)
    ⊢ Eq ((Homotopy.nullHomotopicMap' h).f k₁) (CategoryTheory.CategoryStruct.comp …
  -/
  simp only [nullHomotopicMap']
  /-
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    c : ComplexShape ι
    C D : HomologicalComplex V c
    k₁ k₀ : ι
    r₁₀ : c.Rel k₁ k₀
    hk₁ : ∀ (l : ι), Not (c.Rel l k₁)
    h : (i j : ι) → c.Rel j i → Quiver.Hom (C.X i) (D.X j)
    ⊢ Eq ((Homotopy.nullHomotopicMap fun i j => dite (c.Rel j i) (h i j) fun x =>  …
  -/
  rw [nullHomotopicMap_f_of_not_rel_right r₁₀ hk₁]
  /-
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    c : ComplexShape ι
    C D : HomologicalComplex V c
    k₁ k₀ : ι
    r₁₀ : c.Rel k₁ k₀
    hk₁ : ∀ (l : ι), Not (c.Rel l k₁)
    h : (i j : ι) → c.Rel j i → Quiver.Hom (C.X i) (D.X j)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (C.d k₁ k₀) (dite (c.Rel k₁ k₀) (h k₀ …
  -/
  split_ifs
  /-
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    c : ComplexShape ι
    C D : HomologicalComplex V c
    k₁ k₀ : ι
    r₁₀ : c.Rel k₁ k₀
    hk₁ : ∀ (l : ι), Not (c.Rel l k₁)
    h : (i j : ι) → c.Rel j i → Quiver.Hom (C.X i) (D.X j)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (C.d k₁ k₀) (h k₀ k₁ r₁₀)) (CategoryT …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem nullHomotopicMap_f_eq_zero {k₀ : ι} (hk₀ : ∀ l : ι, ¬c.Rel k₀ l)
    (hk₀' : ∀ l : ι, ¬c.Rel l k₀) (hom : ∀ i j, C.X i ⟶ D.X j) :
    (nullHomotopicMap hom).f k₀ = 0 := by
  /-
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    c : ComplexShape ι
    C D : HomologicalComplex V c
    k₀ : ι
    hk₀ : ∀ (l : ι), Not (c.Rel k₀ l)
    hk₀' : ∀ (l : ι), Not (c.Rel l k₀)
    hom : (i j : ι) → Quiver.Hom (C.X i) (D.X j)
    ⊢ Eq ((Homotopy.nullHomotopicMap hom).f k₀) 0
  -/
  dsimp [nullHomotopicMap, dNext, prevD]
  /-
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    c : ComplexShape ι
    C D : HomologicalComplex V c
    k₀ : ι
    hk₀ : ∀ (l : ι), Not (c.Rel k₀ l)
    hk₀' : ∀ (l : ι), Not (c.Rel l k₀)
    hom : (i j : ι) → Quiver.Hom (C.X i) (D.X j)
    ⊢ Eq (HAdd.hAdd (CategoryTheory.CategoryStruct.comp (C.d k₀ (c.next k₀)) (hom  …
  -/
                                                            /-
                                                              🎉 no goals
                                                            -/
  rw [C.shape, D.shape, zero_comp, comp_zero, add_zero] <;> apply_assumption
                                                            /-
                                                              🎉 no goals
                                                            -/


@[simp]
theorem nullHomotopicMap'_f_eq_zero {k₀ : ι} (hk₀ : ∀ l : ι, ¬c.Rel k₀ l)
    (hk₀' : ∀ l : ι, ¬c.Rel l k₀) (h : ∀ i j, c.Rel j i → (C.X i ⟶ D.X j)) :
    (nullHomotopicMap' h).f k₀ = 0 := by
  /-
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    c : ComplexShape ι
    C D : HomologicalComplex V c
    k₀ : ι
    hk₀ : ∀ (l : ι), Not (c.Rel k₀ l)
    hk₀' : ∀ (l : ι), Not (c.Rel l k₀)
    h : (i j : ι) → c.Rel j i → Quiver.Hom (C.X i) (D.X j)
    ⊢ Eq ((Homotopy.nullHomotopicMap' h).f k₀) 0
  -/
  simp only [nullHomotopicMap']
  /-
    ι : Type u_1
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    c : ComplexShape ι
    C D : HomologicalComplex V c
    k₀ : ι
    hk₀ : ∀ (l : ι), Not (c.Rel k₀ l)
    hk₀' : ∀ (l : ι), Not (c.Rel l k₀)
    h : (i j : ι) → c.Rel j i → Quiver.Hom (C.X i) (D.X j)
    ⊢ Eq ((Homotopy.nullHomotopicMap fun i j => dite (c.Rel j i) (h i j) fun x =>  …
  -/
  apply nullHomotopicMap_f_eq_zero hk₀ hk₀'
  /-
    🎉 no goals
  -/


@[simp 1100]
theorem prevD_chainComplex (f : ∀ i j, P.X i ⟶ Q.X j) (j : ℕ) :
    prevD j f = f j (j + 1) ≫ Q.d _ _ := by
  /-
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    P Q : ChainComplex V Nat
    f : (i j : Nat) → Quiver.Hom (P.X i) (Q.X j)
    j : Nat
    ⊢ Eq ((prevD j) f) (CategoryTheory.CategoryStruct.comp (f j (HAdd.hAdd j 1)) ( …
  -/
  dsimp [prevD]
  /-
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    P Q : ChainComplex V Nat
    f : (i j : Nat) → Quiver.Hom (P.X i) (Q.X j)
    j : Nat
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (f j ((ComplexShape.down Nat).prev j) …
  -/
  have : (ComplexShape.down ℕ).prev j = j + 1 := ChainComplex.prev ℕ j
  /-
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    P Q : ChainComplex V Nat
    f : (i j : Nat) → Quiver.Hom (P.X i) (Q.X j)
    j : Nat
    this : Eq ((ComplexShape.down Nat).prev j) (HAdd.hAdd j 1)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (f j ((ComplexShape.down Nat).prev j) …
  -/
  congr 2
  /-
    🎉 no goals
  -/


@[simp 1100]
theorem dNext_succ_chainComplex (f : ∀ i j, P.X i ⟶ Q.X j) (i : ℕ) :
    dNext (i + 1) f = P.d _ _ ≫ f i (i + 1) := by
  /-
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    P Q : ChainComplex V Nat
    f : (i j : Nat) → Quiver.Hom (P.X i) (Q.X j)
    i : Nat
    ⊢ Eq ((dNext (HAdd.hAdd i 1)) f) (CategoryTheory.CategoryStruct.comp (P.d (HAd …
  -/
  dsimp [dNext]
  /-
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    P Q : ChainComplex V Nat
    f : (i j : Nat) → Quiver.Hom (P.X i) (Q.X j)
    i : Nat
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (P.d (HAdd.hAdd i 1) ((ComplexShape.d …
  -/
  have : (ComplexShape.down ℕ).next (i + 1) = i := ChainComplex.next_nat_succ _
  /-
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    P Q : ChainComplex V Nat
    f : (i j : Nat) → Quiver.Hom (P.X i) (Q.X j)
    i : Nat
    this : Eq ((ComplexShape.down Nat).next (HAdd.hAdd i 1)) i
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (P.d (HAdd.hAdd i 1) ((ComplexShape.d …
  -/
  congr 2
  /-
    🎉 no goals
  -/


@[simp 1100]
theorem dNext_zero_chainComplex (f : ∀ i j, P.X i ⟶ Q.X j) : dNext 0 f = 0 := by
  /-
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    P Q : ChainComplex V Nat
    f : (i j : Nat) → Quiver.Hom (P.X i) (Q.X j)
    ⊢ Eq ((dNext 0) f) 0
  -/
  dsimp [dNext]
  /-
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    P Q : ChainComplex V Nat
    f : (i j : Nat) → Quiver.Hom (P.X i) (Q.X j)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (P.d 0 ((ComplexShape.down Nat).next  …
  -/
  rw [P.shape, zero_comp]
  /-
    case a
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    P Q : ChainComplex V Nat
    f : (i j : Nat) → Quiver.Hom (P.X i) (Q.X j)
    ⊢ Not ((ComplexShape.down Nat).Rel 0 ((ComplexShape.down Nat).next 0))
  -/
  rw [ChainComplex.next_nat_zero]; dsimp; decide
                                          /-
                                            🎉 no goals
                                          -/


/-- An auxiliary construction for `mkInductive`.

Here we build by induction a family of diagrams,
but don't require at the type level that these successive diagrams actually agree.
They do in fact agree, and we then capture that at the type level (i.e. by constructing a homotopy)
in `mkInductive`.

At this stage, we don't check the homotopy condition in degree 0,
because it "falls off the end", and is easier to treat using `xNext` and `xPrev`,
which we do in `mkInductiveAux₂`.
-/
@[simp, nolint unusedArguments]
def mkInductiveAux₁ :
    ∀ n,
      Σ' (f : P.X n ⟶ Q.X (n + 1)) (f' : P.X (n + 1) ⟶ Q.X (n + 2)),
        e.f (n + 1) = P.d (n + 1) n ≫ f + f' ≫ Q.d (n + 2) (n + 1)
  | 0 => ⟨zero, one, comm_one⟩
  | 1 => ⟨one, (succ 0 ⟨zero, one, comm_one⟩).1, (succ 0 ⟨zero, one, comm_one⟩).2⟩
  | n + 2 =>
    ⟨(mkInductiveAux₁ (n + 1)).2.1, (succ (n + 1) (mkInductiveAux₁ (n + 1))).1,
      (succ (n + 1) (mkInductiveAux₁ (n + 1))).2⟩


/-- An auxiliary construction for `mkInductive`.
-/
def mkInductiveAux₂ :
    ∀ n, Σ' (f : P.xNext n ⟶ Q.X n) (f' : P.X n ⟶ Q.xPrev n), e.f n = P.dFrom n ≫ f + f' ≫ Q.dTo n
                                             /-
                                               ι : Type u_1
                                               V : Type u
                                               inst✝¹ : CategoryTheory.Category.{v, u} V
                                               inst✝ : CategoryTheory.Preadditive V
                                               c : ComplexShape ι
                                               C D E : HomologicalComplex V c
                                               f g : Quiver.Hom C D
                                               h k : Quiver.Hom D E
                                               i : ι
                                               P Q : ChainComplex V Nat
                                               e : Quiver.Hom P Q
                                               zero : Quiver.Hom (P.X 0) (Q.X 1)
                                               comm_zero : Eq (e.f 0) (CategoryTheory.CategoryStruct.comp zero (Q.d 1 0))
                                               one : Quiver.Hom (P.X 1) (Q.X 2)
                                               comm_one : Eq (e.f 1) (HAdd.hAdd (CategoryTheory.CategoryStruct.comp (P.d 1 0) …
                                               succ : (n : Nat) → (p : PSigma fun f => PSigma fun f' => Eq (e.f (HAdd.hAdd n  …
                                               ⊢ Eq (e.f 0) (HAdd.hAdd (CategoryTheory.CategoryStruct.comp (HomologicalComple …
                                             -/
  | 0 => ⟨0, zero ≫ (Q.xPrevIso rfl).inv, by simpa using comm_zero⟩
                                             /-
                                               🎉 no goals
                                             -/
  | n + 1 =>
    let I := mkInductiveAux₁ e zero --comm_zero
      one comm_one succ n
                                                                  /-
                                                                    ι : Type u_1
                                                                    V : Type u
                                                                    inst✝¹ : CategoryTheory.Category.{v, u} V
                                                                    inst✝ : CategoryTheory.Preadditive V
                                                                    c : ComplexShape ι
                                                                    C D E : HomologicalComplex V c
                                                                    f g : Quiver.Hom C D
                                                                    h k : Quiver.Hom D E
                                                                    i : ι
                                                                    P Q : ChainComplex V Nat
                                                                    e : Quiver.Hom P Q
                                                                    zero : Quiver.Hom (P.X 0) (Q.X 1)
                                                                    comm_zero : Eq (e.f 0) (CategoryTheory.CategoryStruct.comp zero (Q.d 1 0))
                                                                    one : Quiver.Hom (P.X 1) (Q.X 2)
                                                                    comm_one : Eq (e.f 1) (HAdd.hAdd (CategoryTheory.CategoryStruct.comp (P.d 1 0) …
                                                                    succ : (n : Nat) → (p : PSigma fun f => PSigma fun f' => Eq (e.f (HAdd.hAdd n  …
                                                                    n : Nat
                                                                    I : PSigma fun f => PSigma fun f' => Eq (e.f (HAdd.hAdd n 1)) (HAdd.hAdd (Cate …
                                                                    ⊢ Eq (e.f (HAdd.hAdd n 1)) (HAdd.hAdd (CategoryTheory.CategoryStruct.comp (Hom …
                                                                  -/
    ⟨(P.xNextIso rfl).hom ≫ I.1, I.2.1 ≫ (Q.xPrevIso rfl).inv, by simpa using I.2.2⟩
                                                                  /-
                                                                    🎉 no goals
                                                                  -/

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11647): during the port we marked these lemmas
-- with `@[eqns]` to emulate the old Lean 3 behaviour.


@[simp] theorem mkInductiveAux₂_zero :
    mkInductiveAux₂ e zero comm_zero one comm_one succ 0 =
      ⟨0, zero ≫ (Q.xPrevIso rfl).inv, mkInductiveAux₂.proof_2 e zero comm_zero⟩ :=
  rfl


@[simp] theorem mkInductiveAux₂_add_one (n) :
    mkInductiveAux₂ e zero comm_zero one comm_one succ (n + 1) =
      let I := mkInductiveAux₁ e zero one comm_one succ n
      ⟨(P.xNextIso rfl).hom ≫ I.1, I.2.1 ≫ (Q.xPrevIso rfl).inv,
        mkInductiveAux₂.proof_5 e zero one comm_one succ n⟩ :=
  rfl


theorem mkInductiveAux₃ (i j : ℕ) (h : i + 1 = j) :
    (mkInductiveAux₂ e zero comm_zero one comm_one succ i).2.1 ≫ (Q.xPrevIso h).hom =
      (P.xNextIso h).inv ≫ (mkInductiveAux₂ e zero comm_zero one comm_one succ j).1 := by
  /-
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    P Q : ChainComplex V Nat
    e : Quiver.Hom P Q
    zero : Quiver.Hom (P.X 0) (Q.X 1)
    comm_zero : Eq (e.f 0) (CategoryTheory.CategoryStruct.comp zero (Q.d 1 0))
    one : Quiver.Hom (P.X 1) (Q.X 2)
    comm_one : Eq (e.f 1) (HAdd.hAdd (CategoryTheory.CategoryStruct.comp (P.d 1 0) …
    succ : (n : Nat) → (p : PSigma fun f => PSigma fun f' => Eq (e.f (HAdd.hAdd n  …
    i j : Nat
    h : Eq (HAdd.hAdd i 1) j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (Homotopy.mkInductiveAux₂ e zero comm …
  -/
  subst j
  /-
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    P Q : ChainComplex V Nat
    e : Quiver.Hom P Q
    zero : Quiver.Hom (P.X 0) (Q.X 1)
    comm_zero : Eq (e.f 0) (CategoryTheory.CategoryStruct.comp zero (Q.d 1 0))
    one : Quiver.Hom (P.X 1) (Q.X 2)
    comm_one : Eq (e.f 1) (HAdd.hAdd (CategoryTheory.CategoryStruct.comp (P.d 1 0) …
    succ : (n : Nat) → (p : PSigma fun f => PSigma fun f' => Eq (e.f (HAdd.hAdd n  …
    i : Nat
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (Homotopy.mkInductiveAux₂ e zero comm …
  -/
                                /-
                                  🎉 no goals
                                -/
                                /-
                                  🎉 no goals
                                -/
  rcases i with (_ | _ | i) <;> simp [mkInductiveAux₂]
                                /-
                                  🎉 no goals
                                -/


/-- A constructor for a `Homotopy e 0`, for `e` a chain map between `ℕ`-indexed chain complexes,
working by induction.

You need to provide the components of the homotopy in degrees 0 and 1,
show that these satisfy the homotopy condition,
and then give a construction of each component,
and the fact that it satisfies the homotopy condition,
using as an inductive hypothesis the data and homotopy condition for the previous two components.
-/
def mkInductive : Homotopy e 0 where
  hom i j :=
    if h : i + 1 = j then
      (mkInductiveAux₂ e zero comm_zero one comm_one succ i).2.1 ≫ (Q.xPrevIso h).hom
    else 0
                   /-
                     ι : Type u_1
                     V : Type u
                     inst✝¹ : CategoryTheory.Category.{v, u} V
                     inst✝ : CategoryTheory.Preadditive V
                     c : ComplexShape ι
                     C D E : HomologicalComplex V c
                     f g : Quiver.Hom C D
                     h k : Quiver.Hom D E
                     i✝ : ι
                     P Q : ChainComplex V Nat
                     e : Quiver.Hom P Q
                     zero : Quiver.Hom (P.X 0) (Q.X 1)
                     comm_zero : Eq (e.f 0) (CategoryTheory.CategoryStruct.comp zero (Q.d 1 0))
                     one : Quiver.Hom (P.X 1) (Q.X 2)
                     comm_one : Eq (e.f 1) (HAdd.hAdd (CategoryTheory.CategoryStruct.comp (P.d 1 0) …
                     succ : (n : Nat) → (p : PSigma fun f => PSigma fun f' => Eq (e.f (HAdd.hAdd n  …
                     i j : Nat
                     w : Not ((ComplexShape.down Nat).Rel j i)
                     ⊢ Eq ((fun i j => dite (Eq (HAdd.hAdd i 1) j) (fun h => CategoryTheory.Categor …
                   -/
  zero i j w := by dsimp; rw [dif_neg]; exact w
                                        /-
                                          🎉 no goals
                                        -/
  comm i := by
    /-
      ι : Type u_1
      V : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} V
      inst✝ : CategoryTheory.Preadditive V
      c : ComplexShape ι
      C D E : HomologicalComplex V c
      f g : Quiver.Hom C D
      h k : Quiver.Hom D E
      i✝ : ι
      P Q : ChainComplex V Nat
      e : Quiver.Hom P Q
      zero : Quiver.Hom (P.X 0) (Q.X 1)
      comm_zero : Eq (e.f 0) (CategoryTheory.CategoryStruct.comp zero (Q.d 1 0))
      one : Quiver.Hom (P.X 1) (Q.X 2)
      comm_one : Eq (e.f 1) (HAdd.hAdd (CategoryTheory.CategoryStruct.comp (P.d 1 0) …
      succ : (n : Nat) → (p : PSigma fun f => PSigma fun f' => Eq (e.f (HAdd.hAdd n  …
      i : Nat
      ⊢ Eq (e.f i) (HAdd.hAdd (HAdd.hAdd ((dNext i) fun i j => dite (Eq (HAdd.hAdd i …
    -/
    dsimp
    /-
      ι : Type u_1
      V : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} V
      inst✝ : CategoryTheory.Preadditive V
      c : ComplexShape ι
      C D E : HomologicalComplex V c
      f g : Quiver.Hom C D
      h k : Quiver.Hom D E
      i✝ : ι
      P Q : ChainComplex V Nat
      e : Quiver.Hom P Q
      zero : Quiver.Hom (P.X 0) (Q.X 1)
      comm_zero : Eq (e.f 0) (CategoryTheory.CategoryStruct.comp zero (Q.d 1 0))
      one : Quiver.Hom (P.X 1) (Q.X 2)
      comm_one : Eq (e.f 1) (HAdd.hAdd (CategoryTheory.CategoryStruct.comp (P.d 1 0) …
      succ : (n : Nat) → (p : PSigma fun f => PSigma fun f' => Eq (e.f (HAdd.hAdd n  …
      i : Nat
      ⊢ Eq (e.f i) (HAdd.hAdd (HAdd.hAdd (CategoryTheory.CategoryStruct.comp (Homolo …
    -/
    simp only [add_zero]
    /-
      ι : Type u_1
      V : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} V
      inst✝ : CategoryTheory.Preadditive V
      c : ComplexShape ι
      C D E : HomologicalComplex V c
      f g : Quiver.Hom C D
      h k : Quiver.Hom D E
      i✝ : ι
      P Q : ChainComplex V Nat
      e : Quiver.Hom P Q
      zero : Quiver.Hom (P.X 0) (Q.X 1)
      comm_zero : Eq (e.f 0) (CategoryTheory.CategoryStruct.comp zero (Q.d 1 0))
      one : Quiver.Hom (P.X 1) (Q.X 2)
      comm_one : Eq (e.f 1) (HAdd.hAdd (CategoryTheory.CategoryStruct.comp (P.d 1 0) …
      succ : (n : Nat) → (p : PSigma fun f => PSigma fun f' => Eq (e.f (HAdd.hAdd n  …
      i : Nat
      ⊢ Eq (e.f i) (HAdd.hAdd (CategoryTheory.CategoryStruct.comp (HomologicalComple …
    -/
    refine (mkInductiveAux₂ e zero comm_zero one comm_one succ i).2.2.trans ?_
    /-
      ι : Type u_1
      V : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} V
      inst✝ : CategoryTheory.Preadditive V
      c : ComplexShape ι
      C D E : HomologicalComplex V c
      f g : Quiver.Hom C D
      h k : Quiver.Hom D E
      i✝ : ι
      P Q : ChainComplex V Nat
      e : Quiver.Hom P Q
      zero : Quiver.Hom (P.X 0) (Q.X 1)
      comm_zero : Eq (e.f 0) (CategoryTheory.CategoryStruct.comp zero (Q.d 1 0))
      one : Quiver.Hom (P.X 1) (Q.X 2)
      comm_one : Eq (e.f 1) (HAdd.hAdd (CategoryTheory.CategoryStruct.comp (P.d 1 0) …
      succ : (n : Nat) → (p : PSigma fun f => PSigma fun f' => Eq (e.f (HAdd.hAdd n  …
      i : Nat
      ⊢ Eq (HAdd.hAdd (CategoryTheory.CategoryStruct.comp (HomologicalComplex.dFrom  …
    -/
    congr
      /-
        case e_a.e_a
        ι : Type u_1
        V : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} V
        inst✝ : CategoryTheory.Preadditive V
        c : ComplexShape ι
        C D E : HomologicalComplex V c
        f g : Quiver.Hom C D
        h k : Quiver.Hom D E
        i✝ : ι
        P Q : ChainComplex V Nat
        e : Quiver.Hom P Q
        zero : Quiver.Hom (P.X 0) (Q.X 1)
        comm_zero : Eq (e.f 0) (CategoryTheory.CategoryStruct.comp zero (Q.d 1 0))
        one : Quiver.Hom (P.X 1) (Q.X 2)
        comm_one : Eq (e.f 1) (HAdd.hAdd (CategoryTheory.CategoryStruct.comp (P.d 1 0) …
        succ : (n : Nat) → (p : PSigma fun f => PSigma fun f' => Eq (e.f (HAdd.hAdd n  …
        i : Nat
        ⊢ Eq (Homotopy.mkInductiveAux₂ e zero comm_zero one comm_one succ i).fst ((fro …
      -/
    · cases i
        /-
          case e_a.e_a.zero
          ι : Type u_1
          V : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} V
          inst✝ : CategoryTheory.Preadditive V
          c : ComplexShape ι
          C D E : HomologicalComplex V c
          f g : Quiver.Hom C D
          h k : Quiver.Hom D E
          i : ι
          P Q : ChainComplex V Nat
          e : Quiver.Hom P Q
          zero : Quiver.Hom (P.X 0) (Q.X 1)
          comm_zero : Eq (e.f 0) (CategoryTheory.CategoryStruct.comp zero (Q.d 1 0))
          one : Quiver.Hom (P.X 1) (Q.X 2)
          comm_one : Eq (e.f 1) (HAdd.hAdd (CategoryTheory.CategoryStruct.comp (P.d 1 0) …
          succ : (n : Nat) → (p : PSigma fun f => PSigma fun f' => Eq (e.f (HAdd.hAdd n  …
          ⊢ Eq (Homotopy.mkInductiveAux₂ e zero comm_zero one comm_one succ 0).fst ((fro …
        -/
      · dsimp [fromNext, mkInductiveAux₂]
        /-
          🎉 no goals
        -/
        /-
          case e_a.e_a.succ
          ι : Type u_1
          V : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} V
          inst✝ : CategoryTheory.Preadditive V
          c : ComplexShape ι
          C D E : HomologicalComplex V c
          f g : Quiver.Hom C D
          h k : Quiver.Hom D E
          i : ι
          P Q : ChainComplex V Nat
          e : Quiver.Hom P Q
          zero : Quiver.Hom (P.X 0) (Q.X 1)
          comm_zero : Eq (e.f 0) (CategoryTheory.CategoryStruct.comp zero (Q.d 1 0))
          one : Quiver.Hom (P.X 1) (Q.X 2)
          comm_one : Eq (e.f 1) (HAdd.hAdd (CategoryTheory.CategoryStruct.comp (P.d 1 0) …
          succ : (n : Nat) → (p : PSigma fun f => PSigma fun f' => Eq (e.f (HAdd.hAdd n  …
          n✝ : Nat
          ⊢ Eq (Homotopy.mkInductiveAux₂ e zero comm_zero one comm_one succ (HAdd.hAdd n …
        -/
      · dsimp [fromNext]
        /-
          case e_a.e_a.succ
          ι : Type u_1
          V : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} V
          inst✝ : CategoryTheory.Preadditive V
          c : ComplexShape ι
          C D E : HomologicalComplex V c
          f g : Quiver.Hom C D
          h k : Quiver.Hom D E
          i : ι
          P Q : ChainComplex V Nat
          e : Quiver.Hom P Q
          zero : Quiver.Hom (P.X 0) (Q.X 1)
          comm_zero : Eq (e.f 0) (CategoryTheory.CategoryStruct.comp zero (Q.d 1 0))
          one : Quiver.Hom (P.X 1) (Q.X 2)
          comm_one : Eq (e.f 1) (HAdd.hAdd (CategoryTheory.CategoryStruct.comp (P.d 1 0) …
          succ : (n : Nat) → (p : PSigma fun f => PSigma fun f' => Eq (e.f (HAdd.hAdd n  …
          n✝ : Nat
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.xNextIso P ⋯).hom …
        -/
        simp only [ChainComplex.next_nat_succ, dite_true]
        /-
          case e_a.e_a.succ
          ι : Type u_1
          V : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} V
          inst✝ : CategoryTheory.Preadditive V
          c : ComplexShape ι
          C D E : HomologicalComplex V c
          f g : Quiver.Hom C D
          h k : Quiver.Hom D E
          i : ι
          P Q : ChainComplex V Nat
          e : Quiver.Hom P Q
          zero : Quiver.Hom (P.X 0) (Q.X 1)
          comm_zero : Eq (e.f 0) (CategoryTheory.CategoryStruct.comp zero (Q.d 1 0))
          one : Quiver.Hom (P.X 1) (Q.X 2)
          comm_one : Eq (e.f 1) (HAdd.hAdd (CategoryTheory.CategoryStruct.comp (P.d 1 0) …
          succ : (n : Nat) → (p : PSigma fun f => PSigma fun f' => Eq (e.f (HAdd.hAdd n  …
          n✝ : Nat
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.xNextIso P ⋯).hom …
        -/
        rw [mkInductiveAux₃ e zero comm_zero one comm_one succ]
        /-
          case e_a.e_a.succ
          ι : Type u_1
          V : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} V
          inst✝ : CategoryTheory.Preadditive V
          c : ComplexShape ι
          C D E : HomologicalComplex V c
          f g : Quiver.Hom C D
          h k : Quiver.Hom D E
          i : ι
          P Q : ChainComplex V Nat
          e : Quiver.Hom P Q
          zero : Quiver.Hom (P.X 0) (Q.X 1)
          comm_zero : Eq (e.f 0) (CategoryTheory.CategoryStruct.comp zero (Q.d 1 0))
          one : Quiver.Hom (P.X 1) (Q.X 2)
          comm_one : Eq (e.f 1) (HAdd.hAdd (CategoryTheory.CategoryStruct.comp (P.d 1 0) …
          succ : (n : Nat) → (p : PSigma fun f => PSigma fun f' => Eq (e.f (HAdd.hAdd n  …
          n✝ : Nat
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.xNextIso P ⋯).hom …
        -/
        dsimp [xNextIso]
        /-
          case e_a.e_a.succ
          ι : Type u_1
          V : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} V
          inst✝ : CategoryTheory.Preadditive V
          c : ComplexShape ι
          C D E : HomologicalComplex V c
          f g : Quiver.Hom C D
          h k : Quiver.Hom D E
          i : ι
          P Q : ChainComplex V Nat
          e : Quiver.Hom P Q
          zero : Quiver.Hom (P.X 0) (Q.X 1)
          comm_zero : Eq (e.f 0) (CategoryTheory.CategoryStruct.comp zero (Q.d 1 0))
          one : Quiver.Hom (P.X 1) (Q.X 2)
          comm_one : Eq (e.f 1) (HAdd.hAdd (CategoryTheory.CategoryStruct.comp (P.d 1 0) …
          succ : (n : Nat) → (p : PSigma fun f => PSigma fun f' => Eq (e.f (HAdd.hAdd n  …
          n✝ : Nat
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (Homotopy. …
        -/
        rw [id_comp]
        /-
          🎉 no goals
        -/
      /-
        case e_a.e_a
        ι : Type u_1
        V : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} V
        inst✝ : CategoryTheory.Preadditive V
        c : ComplexShape ι
        C D E : HomologicalComplex V c
        f g : Quiver.Hom C D
        h k : Quiver.Hom D E
        i✝ : ι
        P Q : ChainComplex V Nat
        e : Quiver.Hom P Q
        zero : Quiver.Hom (P.X 0) (Q.X 1)
        comm_zero : Eq (e.f 0) (CategoryTheory.CategoryStruct.comp zero (Q.d 1 0))
        one : Quiver.Hom (P.X 1) (Q.X 2)
        comm_one : Eq (e.f 1) (HAdd.hAdd (CategoryTheory.CategoryStruct.comp (P.d 1 0) …
        succ : (n : Nat) → (p : PSigma fun f => PSigma fun f' => Eq (e.f (HAdd.hAdd n  …
        i : Nat
        ⊢ Eq (Homotopy.mkInductiveAux₂ e zero comm_zero one comm_one succ i).snd.fst ( …
      -/
    · dsimp [toPrev]
      /-
        case e_a.e_a
        ι : Type u_1
        V : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} V
        inst✝ : CategoryTheory.Preadditive V
        c : ComplexShape ι
        C D E : HomologicalComplex V c
        f g : Quiver.Hom C D
        h k : Quiver.Hom D E
        i✝ : ι
        P Q : ChainComplex V Nat
        e : Quiver.Hom P Q
        zero : Quiver.Hom (P.X 0) (Q.X 1)
        comm_zero : Eq (e.f 0) (CategoryTheory.CategoryStruct.comp zero (Q.d 1 0))
        one : Quiver.Hom (P.X 1) (Q.X 2)
        comm_one : Eq (e.f 1) (HAdd.hAdd (CategoryTheory.CategoryStruct.comp (P.d 1 0) …
        succ : (n : Nat) → (p : PSigma fun f => PSigma fun f' => Eq (e.f (HAdd.hAdd n  …
        i : Nat
        ⊢ Eq (Homotopy.mkInductiveAux₂ e zero comm_zero one comm_one succ i).snd.fst ( …
      -/
      erw [dif_pos, comp_id]
      /-
        case e_a.e_a.hc
        ι : Type u_1
        V : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} V
        inst✝ : CategoryTheory.Preadditive V
        c : ComplexShape ι
        C D E : HomologicalComplex V c
        f g : Quiver.Hom C D
        h k : Quiver.Hom D E
        i✝ : ι
        P Q : ChainComplex V Nat
        e : Quiver.Hom P Q
        zero : Quiver.Hom (P.X 0) (Q.X 1)
        comm_zero : Eq (e.f 0) (CategoryTheory.CategoryStruct.comp zero (Q.d 1 0))
        one : Quiver.Hom (P.X 1) (Q.X 2)
        comm_one : Eq (e.f 1) (HAdd.hAdd (CategoryTheory.CategoryStruct.comp (P.d 1 0) …
        succ : (n : Nat) → (p : PSigma fun f => PSigma fun f' => Eq (e.f (HAdd.hAdd n  …
        i : Nat
        ⊢ Eq (HAdd.hAdd i 1) ((ComplexShape.down Nat).prev i)
      -/
      simp only [ChainComplex.prev]
      /-
        🎉 no goals
      -/


@[simp 1100]
theorem dNext_cochainComplex (f : ∀ i j, P.X i ⟶ Q.X j) (j : ℕ) :
    dNext j f = P.d _ _ ≫ f (j + 1) j := by
  /-
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    P Q : CochainComplex V Nat
    f : (i j : Nat) → Quiver.Hom (P.X i) (Q.X j)
    j : Nat
    ⊢ Eq ((dNext j) f) (CategoryTheory.CategoryStruct.comp (P.d j (HAdd.hAdd j 1)) …
  -/
  dsimp [dNext]
  /-
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    P Q : CochainComplex V Nat
    f : (i j : Nat) → Quiver.Hom (P.X i) (Q.X j)
    j : Nat
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (P.d j ((ComplexShape.up Nat).next j) …
  -/
  have : (ComplexShape.up ℕ).next j = j + 1 := CochainComplex.next ℕ j
  /-
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    P Q : CochainComplex V Nat
    f : (i j : Nat) → Quiver.Hom (P.X i) (Q.X j)
    j : Nat
    this : Eq ((ComplexShape.up Nat).next j) (HAdd.hAdd j 1)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (P.d j ((ComplexShape.up Nat).next j) …
  -/
  congr 2
  /-
    🎉 no goals
  -/


@[simp 1100]
theorem prevD_succ_cochainComplex (f : ∀ i j, P.X i ⟶ Q.X j) (i : ℕ) :
    prevD (i + 1) f = f (i + 1) _ ≫ Q.d i (i + 1) := by
  /-
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    P Q : CochainComplex V Nat
    f : (i j : Nat) → Quiver.Hom (P.X i) (Q.X j)
    i : Nat
    ⊢ Eq ((prevD (HAdd.hAdd i 1)) f) (CategoryTheory.CategoryStruct.comp (f (HAdd. …
  -/
  dsimp [prevD]
  /-
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    P Q : CochainComplex V Nat
    f : (i j : Nat) → Quiver.Hom (P.X i) (Q.X j)
    i : Nat
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (f (HAdd.hAdd i 1) ((ComplexShape.up  …
  -/
  have : (ComplexShape.up ℕ).prev (i + 1) = i := CochainComplex.prev_nat_succ i
  /-
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    P Q : CochainComplex V Nat
    f : (i j : Nat) → Quiver.Hom (P.X i) (Q.X j)
    i : Nat
    this : Eq ((ComplexShape.up Nat).prev (HAdd.hAdd i 1)) i
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (f (HAdd.hAdd i 1) ((ComplexShape.up  …
  -/
  congr 2
  /-
    🎉 no goals
  -/


@[simp 1100]
theorem prevD_zero_cochainComplex (f : ∀ i j, P.X i ⟶ Q.X j) : prevD 0 f = 0 := by
  /-
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    P Q : CochainComplex V Nat
    f : (i j : Nat) → Quiver.Hom (P.X i) (Q.X j)
    ⊢ Eq ((prevD 0) f) 0
  -/
  dsimp [prevD]
  /-
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    P Q : CochainComplex V Nat
    f : (i j : Nat) → Quiver.Hom (P.X i) (Q.X j)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (f 0 ((ComplexShape.up Nat).prev 0))  …
  -/
  rw [Q.shape, comp_zero]
  /-
    case a
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    P Q : CochainComplex V Nat
    f : (i j : Nat) → Quiver.Hom (P.X i) (Q.X j)
    ⊢ Not ((ComplexShape.up Nat).Rel ((ComplexShape.up Nat).prev 0) 0)
  -/
  rw [CochainComplex.prev_nat_zero]; dsimp; decide
                                            /-
                                              🎉 no goals
                                            -/


/-- An auxiliary construction for `mkCoinductive`.

Here we build by induction a family of diagrams,
but don't require at the type level that these successive diagrams actually agree.
They do in fact agree, and we then capture that at the type level (i.e. by constructing a homotopy)
in `mkCoinductive`.

At this stage, we don't check the homotopy condition in degree 0,
because it "falls off the end", and is easier to treat using `xNext` and `xPrev`,
which we do in `mkInductiveAux₂`.
-/
@[simp]
def mkCoinductiveAux₁ :
    ∀ n,
      Σ' (f : P.X (n + 1) ⟶ Q.X n) (f' : P.X (n + 2) ⟶ Q.X (n + 1)),
        e.f (n + 1) = f ≫ Q.d n (n + 1) + P.d (n + 1) (n + 2) ≫ f'
  | 0 => ⟨zero, one, comm_one⟩
  | 1 => ⟨one, (succ 0 ⟨zero, one, comm_one⟩).1, (succ 0 ⟨zero, one, comm_one⟩).2⟩
  | n + 2 =>
    ⟨(mkCoinductiveAux₁ (n + 1)).2.1, (succ (n + 1) (mkCoinductiveAux₁ (n + 1))).1,
      (succ (n + 1) (mkCoinductiveAux₁ (n + 1))).2⟩


/-- An auxiliary construction for `mkInductive`.
-/
def mkCoinductiveAux₂ :
    ∀ n, Σ' (f : P.X n ⟶ Q.xPrev n) (f' : P.xNext n ⟶ Q.X n), e.f n = f ≫ Q.dTo n + P.dFrom n ≫ f'
                                             /-
                                               ι : Type u_1
                                               V : Type u
                                               inst✝¹ : CategoryTheory.Category.{v, u} V
                                               inst✝ : CategoryTheory.Preadditive V
                                               c : ComplexShape ι
                                               C D E : HomologicalComplex V c
                                               f g : Quiver.Hom C D
                                               h k : Quiver.Hom D E
                                               i : ι
                                               P Q : CochainComplex V Nat
                                               e : Quiver.Hom P Q
                                               zero : Quiver.Hom (P.X 1) (Q.X 0)
                                               comm_zero : Eq (e.f 0) (CategoryTheory.CategoryStruct.comp (P.d 0 1) zero)
                                               one : Quiver.Hom (P.X 2) (Q.X 1)
                                               comm_one : Eq (e.f 1) (HAdd.hAdd (CategoryTheory.CategoryStruct.comp zero (Q.d …
                                               succ : (n : Nat) → (p : PSigma fun f => PSigma fun f' => Eq (e.f (HAdd.hAdd n  …
                                               ⊢ Eq (e.f 0) (HAdd.hAdd (CategoryTheory.CategoryStruct.comp 0 (HomologicalComp …
                                             -/
  | 0 => ⟨0, (P.xNextIso rfl).hom ≫ zero, by simpa using comm_zero⟩
                                             /-
                                               🎉 no goals
                                             -/
  | n + 1 =>
    let I := mkCoinductiveAux₁ e zero one comm_one succ n
                                                                  /-
                                                                    ι : Type u_1
                                                                    V : Type u
                                                                    inst✝¹ : CategoryTheory.Category.{v, u} V
                                                                    inst✝ : CategoryTheory.Preadditive V
                                                                    c : ComplexShape ι
                                                                    C D E : HomologicalComplex V c
                                                                    f g : Quiver.Hom C D
                                                                    h k : Quiver.Hom D E
                                                                    i : ι
                                                                    P Q : CochainComplex V Nat
                                                                    e : Quiver.Hom P Q
                                                                    zero : Quiver.Hom (P.X 1) (Q.X 0)
                                                                    comm_zero : Eq (e.f 0) (CategoryTheory.CategoryStruct.comp (P.d 0 1) zero)
                                                                    one : Quiver.Hom (P.X 2) (Q.X 1)
                                                                    comm_one : Eq (e.f 1) (HAdd.hAdd (CategoryTheory.CategoryStruct.comp zero (Q.d …
                                                                    succ : (n : Nat) → (p : PSigma fun f => PSigma fun f' => Eq (e.f (HAdd.hAdd n  …
                                                                    n : Nat
                                                                    I : PSigma fun f => PSigma fun f' => Eq (e.f (HAdd.hAdd n 1)) (HAdd.hAdd (Cate …
                                                                    ⊢ Eq (e.f (HAdd.hAdd n 1)) (HAdd.hAdd (CategoryTheory.CategoryStruct.comp (Cat …
                                                                  -/
    ⟨I.1 ≫ (Q.xPrevIso rfl).inv, (P.xNextIso rfl).hom ≫ I.2.1, by simpa using I.2.2⟩
                                                                  /-
                                                                    🎉 no goals
                                                                  -/

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11647): during the port we marked these lemmas with `@[eqns]`
-- to emulate the old Lean 3 behaviour.


@[simp] theorem mkCoinductiveAux₂_zero :
    mkCoinductiveAux₂ e zero comm_zero one comm_one succ 0 =
      ⟨0, (P.xNextIso rfl).hom ≫ zero, mkCoinductiveAux₂.proof_2 e zero comm_zero⟩ :=
  rfl


@[simp] theorem mkCoinductiveAux₂_add_one (n) :
    mkCoinductiveAux₂ e zero comm_zero one comm_one succ (n + 1) =
      let I := mkCoinductiveAux₁ e zero one comm_one succ n
      ⟨I.1 ≫ (Q.xPrevIso rfl).inv, (P.xNextIso rfl).hom ≫ I.2.1,
        mkCoinductiveAux₂.proof_5 e zero one comm_one succ n⟩ :=
  rfl


theorem mkCoinductiveAux₃ (i j : ℕ) (h : i + 1 = j) :
    (P.xNextIso h).inv ≫ (mkCoinductiveAux₂ e zero comm_zero one comm_one succ i).2.1 =
      (mkCoinductiveAux₂ e zero comm_zero one comm_one succ j).1 ≫ (Q.xPrevIso h).hom := by
  /-
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    P Q : CochainComplex V Nat
    e : Quiver.Hom P Q
    zero : Quiver.Hom (P.X 1) (Q.X 0)
    comm_zero : Eq (e.f 0) (CategoryTheory.CategoryStruct.comp (P.d 0 1) zero)
    one : Quiver.Hom (P.X 2) (Q.X 1)
    comm_one : Eq (e.f 1) (HAdd.hAdd (CategoryTheory.CategoryStruct.comp zero (Q.d …
    succ : (n : Nat) → (p : PSigma fun f => PSigma fun f' => Eq (e.f (HAdd.hAdd n  …
    i j : Nat
    h : Eq (HAdd.hAdd i 1) j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.xNextIso P h).inv …
  -/
  subst j
  /-
    V : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} V
    inst✝ : CategoryTheory.Preadditive V
    P Q : CochainComplex V Nat
    e : Quiver.Hom P Q
    zero : Quiver.Hom (P.X 1) (Q.X 0)
    comm_zero : Eq (e.f 0) (CategoryTheory.CategoryStruct.comp (P.d 0 1) zero)
    one : Quiver.Hom (P.X 2) (Q.X 1)
    comm_one : Eq (e.f 1) (HAdd.hAdd (CategoryTheory.CategoryStruct.comp zero (Q.d …
    succ : (n : Nat) → (p : PSigma fun f => PSigma fun f' => Eq (e.f (HAdd.hAdd n  …
    i : Nat
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.xNextIso P ⋯).inv …
  -/
                                /-
                                  🎉 no goals
                                -/
                                /-
                                  🎉 no goals
                                -/
  rcases i with (_ | _ | i) <;> simp [mkCoinductiveAux₂]
                                /-
                                  🎉 no goals
                                -/


/-- A constructor for a `Homotopy e 0`, for `e` a chain map between `ℕ`-indexed cochain complexes,
working by induction.

You need to provide the components of the homotopy in degrees 0 and 1,
show that these satisfy the homotopy condition,
and then give a construction of each component,
and the fact that it satisfies the homotopy condition,
using as an inductive hypothesis the data and homotopy condition for the previous two components.
-/
def mkCoinductive : Homotopy e 0 where
  hom i j :=
    if h : j + 1 = i then
      (P.xNextIso h).inv ≫ (mkCoinductiveAux₂ e zero comm_zero one comm_one succ j).2.1
    else 0
                   /-
                     ι : Type u_1
                     V : Type u
                     inst✝¹ : CategoryTheory.Category.{v, u} V
                     inst✝ : CategoryTheory.Preadditive V
                     c : ComplexShape ι
                     C D E : HomologicalComplex V c
                     f g : Quiver.Hom C D
                     h k : Quiver.Hom D E
                     i✝ : ι
                     P Q : CochainComplex V Nat
                     e : Quiver.Hom P Q
                     zero : Quiver.Hom (P.X 1) (Q.X 0)
                     comm_zero : Eq (e.f 0) (CategoryTheory.CategoryStruct.comp (P.d 0 1) zero)
                     one : Quiver.Hom (P.X 2) (Q.X 1)
                     comm_one : Eq (e.f 1) (HAdd.hAdd (CategoryTheory.CategoryStruct.comp zero (Q.d …
                     succ : (n : Nat) → (p : PSigma fun f => PSigma fun f' => Eq (e.f (HAdd.hAdd n  …
                     i j : Nat
                     w : Not ((ComplexShape.up Nat).Rel j i)
                     ⊢ Eq ((fun i j => dite (Eq (HAdd.hAdd j 1) i) (fun h => CategoryTheory.Categor …
                   -/
  zero i j w := by dsimp; rw [dif_neg]; exact w
                                        /-
                                          🎉 no goals
                                        -/
  comm i := by
    /-
      ι : Type u_1
      V : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} V
      inst✝ : CategoryTheory.Preadditive V
      c : ComplexShape ι
      C D E : HomologicalComplex V c
      f g : Quiver.Hom C D
      h k : Quiver.Hom D E
      i✝ : ι
      P Q : CochainComplex V Nat
      e : Quiver.Hom P Q
      zero : Quiver.Hom (P.X 1) (Q.X 0)
      comm_zero : Eq (e.f 0) (CategoryTheory.CategoryStruct.comp (P.d 0 1) zero)
      one : Quiver.Hom (P.X 2) (Q.X 1)
      comm_one : Eq (e.f 1) (HAdd.hAdd (CategoryTheory.CategoryStruct.comp zero (Q.d …
      succ : (n : Nat) → (p : PSigma fun f => PSigma fun f' => Eq (e.f (HAdd.hAdd n  …
      i : Nat
      ⊢ Eq (e.f i) (HAdd.hAdd (HAdd.hAdd ((dNext i) fun i j => dite (Eq (HAdd.hAdd j …
    -/
    dsimp
    /-
      ι : Type u_1
      V : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} V
      inst✝ : CategoryTheory.Preadditive V
      c : ComplexShape ι
      C D E : HomologicalComplex V c
      f g : Quiver.Hom C D
      h k : Quiver.Hom D E
      i✝ : ι
      P Q : CochainComplex V Nat
      e : Quiver.Hom P Q
      zero : Quiver.Hom (P.X 1) (Q.X 0)
      comm_zero : Eq (e.f 0) (CategoryTheory.CategoryStruct.comp (P.d 0 1) zero)
      one : Quiver.Hom (P.X 2) (Q.X 1)
      comm_one : Eq (e.f 1) (HAdd.hAdd (CategoryTheory.CategoryStruct.comp zero (Q.d …
      succ : (n : Nat) → (p : PSigma fun f => PSigma fun f' => Eq (e.f (HAdd.hAdd n  …
      i : Nat
      ⊢ Eq (e.f i) (HAdd.hAdd (HAdd.hAdd (CategoryTheory.CategoryStruct.comp (Homolo …
    -/
    simp only [add_zero]
    /-
      ι : Type u_1
      V : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} V
      inst✝ : CategoryTheory.Preadditive V
      c : ComplexShape ι
      C D E : HomologicalComplex V c
      f g : Quiver.Hom C D
      h k : Quiver.Hom D E
      i✝ : ι
      P Q : CochainComplex V Nat
      e : Quiver.Hom P Q
      zero : Quiver.Hom (P.X 1) (Q.X 0)
      comm_zero : Eq (e.f 0) (CategoryTheory.CategoryStruct.comp (P.d 0 1) zero)
      one : Quiver.Hom (P.X 2) (Q.X 1)
      comm_one : Eq (e.f 1) (HAdd.hAdd (CategoryTheory.CategoryStruct.comp zero (Q.d …
      succ : (n : Nat) → (p : PSigma fun f => PSigma fun f' => Eq (e.f (HAdd.hAdd n  …
      i : Nat
      ⊢ Eq (e.f i) (HAdd.hAdd (CategoryTheory.CategoryStruct.comp (HomologicalComple …
    -/
    rw [add_comm]
    /-
      ι : Type u_1
      V : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} V
      inst✝ : CategoryTheory.Preadditive V
      c : ComplexShape ι
      C D E : HomologicalComplex V c
      f g : Quiver.Hom C D
      h k : Quiver.Hom D E
      i✝ : ι
      P Q : CochainComplex V Nat
      e : Quiver.Hom P Q
      zero : Quiver.Hom (P.X 1) (Q.X 0)
      comm_zero : Eq (e.f 0) (CategoryTheory.CategoryStruct.comp (P.d 0 1) zero)
      one : Quiver.Hom (P.X 2) (Q.X 1)
      comm_one : Eq (e.f 1) (HAdd.hAdd (CategoryTheory.CategoryStruct.comp zero (Q.d …
      succ : (n : Nat) → (p : PSigma fun f => PSigma fun f' => Eq (e.f (HAdd.hAdd n  …
      i : Nat
      ⊢ Eq (e.f i) (HAdd.hAdd (CategoryTheory.CategoryStruct.comp ((toPrev i) fun i  …
    -/
    refine (mkCoinductiveAux₂ e zero comm_zero one comm_one succ i).2.2.trans ?_
    /-
      ι : Type u_1
      V : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} V
      inst✝ : CategoryTheory.Preadditive V
      c : ComplexShape ι
      C D E : HomologicalComplex V c
      f g : Quiver.Hom C D
      h k : Quiver.Hom D E
      i✝ : ι
      P Q : CochainComplex V Nat
      e : Quiver.Hom P Q
      zero : Quiver.Hom (P.X 1) (Q.X 0)
      comm_zero : Eq (e.f 0) (CategoryTheory.CategoryStruct.comp (P.d 0 1) zero)
      one : Quiver.Hom (P.X 2) (Q.X 1)
      comm_one : Eq (e.f 1) (HAdd.hAdd (CategoryTheory.CategoryStruct.comp zero (Q.d …
      succ : (n : Nat) → (p : PSigma fun f => PSigma fun f' => Eq (e.f (HAdd.hAdd n  …
      i : Nat
      ⊢ Eq (HAdd.hAdd (CategoryTheory.CategoryStruct.comp (Homotopy.mkCoinductiveAux …
    -/
    congr
      /-
        case e_a.e_a
        ι : Type u_1
        V : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} V
        inst✝ : CategoryTheory.Preadditive V
        c : ComplexShape ι
        C D E : HomologicalComplex V c
        f g : Quiver.Hom C D
        h k : Quiver.Hom D E
        i✝ : ι
        P Q : CochainComplex V Nat
        e : Quiver.Hom P Q
        zero : Quiver.Hom (P.X 1) (Q.X 0)
        comm_zero : Eq (e.f 0) (CategoryTheory.CategoryStruct.comp (P.d 0 1) zero)
        one : Quiver.Hom (P.X 2) (Q.X 1)
        comm_one : Eq (e.f 1) (HAdd.hAdd (CategoryTheory.CategoryStruct.comp zero (Q.d …
        succ : (n : Nat) → (p : PSigma fun f => PSigma fun f' => Eq (e.f (HAdd.hAdd n  …
        i : Nat
        ⊢ Eq (Homotopy.mkCoinductiveAux₂ e zero comm_zero one comm_one succ i).fst ((t …
      -/
    · cases i
        /-
          case e_a.e_a.zero
          ι : Type u_1
          V : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} V
          inst✝ : CategoryTheory.Preadditive V
          c : ComplexShape ι
          C D E : HomologicalComplex V c
          f g : Quiver.Hom C D
          h k : Quiver.Hom D E
          i : ι
          P Q : CochainComplex V Nat
          e : Quiver.Hom P Q
          zero : Quiver.Hom (P.X 1) (Q.X 0)
          comm_zero : Eq (e.f 0) (CategoryTheory.CategoryStruct.comp (P.d 0 1) zero)
          one : Quiver.Hom (P.X 2) (Q.X 1)
          comm_one : Eq (e.f 1) (HAdd.hAdd (CategoryTheory.CategoryStruct.comp zero (Q.d …
          succ : (n : Nat) → (p : PSigma fun f => PSigma fun f' => Eq (e.f (HAdd.hAdd n  …
          ⊢ Eq (Homotopy.mkCoinductiveAux₂ e zero comm_zero one comm_one succ 0).fst ((t …
        -/
      · dsimp [toPrev, mkCoinductiveAux₂]
        /-
          🎉 no goals
        -/
        /-
          case e_a.e_a.succ
          ι : Type u_1
          V : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} V
          inst✝ : CategoryTheory.Preadditive V
          c : ComplexShape ι
          C D E : HomologicalComplex V c
          f g : Quiver.Hom C D
          h k : Quiver.Hom D E
          i : ι
          P Q : CochainComplex V Nat
          e : Quiver.Hom P Q
          zero : Quiver.Hom (P.X 1) (Q.X 0)
          comm_zero : Eq (e.f 0) (CategoryTheory.CategoryStruct.comp (P.d 0 1) zero)
          one : Quiver.Hom (P.X 2) (Q.X 1)
          comm_one : Eq (e.f 1) (HAdd.hAdd (CategoryTheory.CategoryStruct.comp zero (Q.d …
          succ : (n : Nat) → (p : PSigma fun f => PSigma fun f' => Eq (e.f (HAdd.hAdd n  …
          n✝ : Nat
          ⊢ Eq (Homotopy.mkCoinductiveAux₂ e zero comm_zero one comm_one succ (HAdd.hAdd …
        -/
      · dsimp [toPrev]
        /-
          case e_a.e_a.succ
          ι : Type u_1
          V : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} V
          inst✝ : CategoryTheory.Preadditive V
          c : ComplexShape ι
          C D E : HomologicalComplex V c
          f g : Quiver.Hom C D
          h k : Quiver.Hom D E
          i : ι
          P Q : CochainComplex V Nat
          e : Quiver.Hom P Q
          zero : Quiver.Hom (P.X 1) (Q.X 0)
          comm_zero : Eq (e.f 0) (CategoryTheory.CategoryStruct.comp (P.d 0 1) zero)
          one : Quiver.Hom (P.X 2) (Q.X 1)
          comm_one : Eq (e.f 1) (HAdd.hAdd (CategoryTheory.CategoryStruct.comp zero (Q.d …
          succ : (n : Nat) → (p : PSigma fun f => PSigma fun f' => Eq (e.f (HAdd.hAdd n  …
          n✝ : Nat
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (Homotopy.mkCoinductiveAux₁ e zero on …
        -/
        simp only [CochainComplex.prev_nat_succ, dite_true]
        /-
          case e_a.e_a.succ
          ι : Type u_1
          V : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} V
          inst✝ : CategoryTheory.Preadditive V
          c : ComplexShape ι
          C D E : HomologicalComplex V c
          f g : Quiver.Hom C D
          h k : Quiver.Hom D E
          i : ι
          P Q : CochainComplex V Nat
          e : Quiver.Hom P Q
          zero : Quiver.Hom (P.X 1) (Q.X 0)
          comm_zero : Eq (e.f 0) (CategoryTheory.CategoryStruct.comp (P.d 0 1) zero)
          one : Quiver.Hom (P.X 2) (Q.X 1)
          comm_one : Eq (e.f 1) (HAdd.hAdd (CategoryTheory.CategoryStruct.comp zero (Q.d …
          succ : (n : Nat) → (p : PSigma fun f => PSigma fun f' => Eq (e.f (HAdd.hAdd n  …
          n✝ : Nat
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (Homotopy.mkCoinductiveAux₁ e zero on …
        -/
        rw [mkCoinductiveAux₃ e zero comm_zero one comm_one succ]
        /-
          case e_a.e_a.succ
          ι : Type u_1
          V : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} V
          inst✝ : CategoryTheory.Preadditive V
          c : ComplexShape ι
          C D E : HomologicalComplex V c
          f g : Quiver.Hom C D
          h k : Quiver.Hom D E
          i : ι
          P Q : CochainComplex V Nat
          e : Quiver.Hom P Q
          zero : Quiver.Hom (P.X 1) (Q.X 0)
          comm_zero : Eq (e.f 0) (CategoryTheory.CategoryStruct.comp (P.d 0 1) zero)
          one : Quiver.Hom (P.X 2) (Q.X 1)
          comm_one : Eq (e.f 1) (HAdd.hAdd (CategoryTheory.CategoryStruct.comp zero (Q.d …
          succ : (n : Nat) → (p : PSigma fun f => PSigma fun f' => Eq (e.f (HAdd.hAdd n  …
          n✝ : Nat
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (Homotopy.mkCoinductiveAux₁ e zero on …
        -/
        dsimp [xPrevIso]
        /-
          case e_a.e_a.succ
          ι : Type u_1
          V : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} V
          inst✝ : CategoryTheory.Preadditive V
          c : ComplexShape ι
          C D E : HomologicalComplex V c
          f g : Quiver.Hom C D
          h k : Quiver.Hom D E
          i : ι
          P Q : CochainComplex V Nat
          e : Quiver.Hom P Q
          zero : Quiver.Hom (P.X 1) (Q.X 0)
          comm_zero : Eq (e.f 0) (CategoryTheory.CategoryStruct.comp (P.d 0 1) zero)
          one : Quiver.Hom (P.X 2) (Q.X 1)
          comm_one : Eq (e.f 1) (HAdd.hAdd (CategoryTheory.CategoryStruct.comp zero (Q.d …
          succ : (n : Nat) → (p : PSigma fun f => PSigma fun f' => Eq (e.f (HAdd.hAdd n  …
          n✝ : Nat
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (Homotopy.mkCoinductiveAux₁ e zero on …
        -/
        rw [comp_id]
        /-
          🎉 no goals
        -/
      /-
        case e_a.e_a
        ι : Type u_1
        V : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} V
        inst✝ : CategoryTheory.Preadditive V
        c : ComplexShape ι
        C D E : HomologicalComplex V c
        f g : Quiver.Hom C D
        h k : Quiver.Hom D E
        i✝ : ι
        P Q : CochainComplex V Nat
        e : Quiver.Hom P Q
        zero : Quiver.Hom (P.X 1) (Q.X 0)
        comm_zero : Eq (e.f 0) (CategoryTheory.CategoryStruct.comp (P.d 0 1) zero)
        one : Quiver.Hom (P.X 2) (Q.X 1)
        comm_one : Eq (e.f 1) (HAdd.hAdd (CategoryTheory.CategoryStruct.comp zero (Q.d …
        succ : (n : Nat) → (p : PSigma fun f => PSigma fun f' => Eq (e.f (HAdd.hAdd n  …
        i : Nat
        ⊢ Eq (Homotopy.mkCoinductiveAux₂ e zero comm_zero one comm_one succ i).snd.fst …
      -/
    · dsimp [fromNext]
      /-
        case e_a.e_a
        ι : Type u_1
        V : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} V
        inst✝ : CategoryTheory.Preadditive V
        c : ComplexShape ι
        C D E : HomologicalComplex V c
        f g : Quiver.Hom C D
        h k : Quiver.Hom D E
        i✝ : ι
        P Q : CochainComplex V Nat
        e : Quiver.Hom P Q
        zero : Quiver.Hom (P.X 1) (Q.X 0)
        comm_zero : Eq (e.f 0) (CategoryTheory.CategoryStruct.comp (P.d 0 1) zero)
        one : Quiver.Hom (P.X 2) (Q.X 1)
        comm_one : Eq (e.f 1) (HAdd.hAdd (CategoryTheory.CategoryStruct.comp zero (Q.d …
        succ : (n : Nat) → (p : PSigma fun f => PSigma fun f' => Eq (e.f (HAdd.hAdd n  …
        i : Nat
        ⊢ Eq (Homotopy.mkCoinductiveAux₂ e zero comm_zero one comm_one succ i).snd.fst …
      -/
      erw [dif_pos, id_comp]
      /-
        case e_a.e_a.hc
        ι : Type u_1
        V : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} V
        inst✝ : CategoryTheory.Preadditive V
        c : ComplexShape ι
        C D E : HomologicalComplex V c
        f g : Quiver.Hom C D
        h k : Quiver.Hom D E
        i✝ : ι
        P Q : CochainComplex V Nat
        e : Quiver.Hom P Q
        zero : Quiver.Hom (P.X 1) (Q.X 0)
        comm_zero : Eq (e.f 0) (CategoryTheory.CategoryStruct.comp (P.d 0 1) zero)
        one : Quiver.Hom (P.X 2) (Q.X 1)
        comm_one : Eq (e.f 1) (HAdd.hAdd (CategoryTheory.CategoryStruct.comp zero (Q.d …
        succ : (n : Nat) → (p : PSigma fun f => PSigma fun f' => Eq (e.f (HAdd.hAdd n  …
        i : Nat
        ⊢ Eq (HAdd.hAdd i 1) ((ComplexShape.up Nat).next i)
      -/
      simp only [CochainComplex.next]
      /-
        🎉 no goals
      -/


/-- A homotopy equivalence between two chain complexes consists of a chain map each way,
and homotopies from the compositions to the identity chain maps.

Note that this contains data;
arguably it might be more useful for many applications if we truncated it to a Prop.
-/
structure HomotopyEquiv (C D : HomologicalComplex V c) where
  hom : C ⟶ D
  inv : D ⟶ C
  homotopyHomInvId : Homotopy (hom ≫ inv) (𝟙 C)
  homotopyInvHomId : Homotopy (inv ≫ hom) (𝟙 D)


variable (V c) in
/-- The morphism property on `HomologicalComplex V c` given by homotopy equivalences. -/
def HomologicalComplex.homotopyEquivalences :
    MorphismProperty (HomologicalComplex V c) :=
  fun X Y f => ∃ (e : HomotopyEquiv X Y), e.hom = f


/-- Any complex is homotopy equivalent to itself. -/
@[refl]
def refl (C : HomologicalComplex V c) : HomotopyEquiv C C where
  hom := 𝟙 C
  inv := 𝟙 C
                                        /-
                                          ι : Type u_1
                                          V : Type u
                                          inst✝¹ : CategoryTheory.Category.{v, u} V
                                          inst✝ : CategoryTheory.Preadditive V
                                          c : ComplexShape ι
                                          C✝ D E : HomologicalComplex V c
                                          f g : Quiver.Hom C✝ D
                                          h k : Quiver.Hom D E
                                          i : ι
                                          C : HomologicalComplex V c
                                          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id C)  …
                                        -/
  homotopyHomInvId := Homotopy.ofEq (by simp)
                                        /-
                                          🎉 no goals
                                        -/
                                        /-
                                          ι : Type u_1
                                          V : Type u
                                          inst✝¹ : CategoryTheory.Category.{v, u} V
                                          inst✝ : CategoryTheory.Preadditive V
                                          c : ComplexShape ι
                                          C✝ D E : HomologicalComplex V c
                                          f g : Quiver.Hom C✝ D
                                          h k : Quiver.Hom D E
                                          i : ι
                                          C : HomologicalComplex V c
                                          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id C)  …
                                        -/
  homotopyInvHomId := Homotopy.ofEq (by simp)
                                        /-
                                          🎉 no goals
                                        -/


instance : Inhabited (HomotopyEquiv C C) :=
  ⟨refl C⟩


/-- Being homotopy equivalent is a symmetric relation. -/
@[symm]
def symm {C D : HomologicalComplex V c} (f : HomotopyEquiv C D) : HomotopyEquiv D C where
  hom := f.inv
  inv := f.hom
  homotopyHomInvId := f.homotopyInvHomId
  homotopyInvHomId := f.homotopyHomInvId


/-- Homotopy equivalence is a transitive relation. -/
@[trans]
def trans {C D E : HomologicalComplex V c} (f : HomotopyEquiv C D) (g : HomotopyEquiv D E) :
    HomotopyEquiv C E where
  hom := f.hom ≫ g.hom
  inv := g.inv ≫ f.inv
  homotopyHomInvId := by simpa using
    ((g.homotopyHomInvId.compRightId f.inv).compLeft f.hom).trans f.homotopyHomInvId
  homotopyInvHomId := by simpa using
    ((f.homotopyInvHomId.compRightId g.hom).compLeft g.inv).trans g.homotopyInvHomId


/-- An isomorphism of complexes induces a homotopy equivalence. -/
def ofIso {ι : Type*} {V : Type u} [Category.{v} V] [Preadditive V] {c : ComplexShape ι}
    {C D : HomologicalComplex V c} (f : C ≅ D) : HomotopyEquiv C D :=
  ⟨f.hom, f.inv, Homotopy.ofEq f.3, Homotopy.ofEq f.4⟩


/-- An additive functor takes homotopies to homotopies. -/
@[simps]
def Functor.mapHomotopy (F : V ⥤ W) [F.Additive] {f g : C ⟶ D} (h : Homotopy f g) :
    Homotopy ((F.mapHomologicalComplex c).map f) ((F.mapHomologicalComplex c).map g) where
  hom i j := F.map (h.hom i j)
                   /-
                     ι : Type u_1
                     V : Type u
                     inst✝⁴ : CategoryTheory.Category.{v, u} V
                     inst✝³ : CategoryTheory.Preadditive V
                     c : ComplexShape ι
                     C D E : HomologicalComplex V c
                     f✝ g✝ : Quiver.Hom C D
                     h✝ k : Quiver.Hom D E
                     i✝ : ι
                     W : Type u_2
                     inst✝² : CategoryTheory.Category.{?u.274677, u_2} W
                     inst✝¹ : CategoryTheory.Preadditive W
                     F : CategoryTheory.Functor V W
                     inst✝ : F.Additive
                     f g : Quiver.Hom C D
                     h : Homotopy f g
                     i j : ι
                     w : Not (c.Rel j i)
                     ⊢ Eq ((fun i j => F.map (h.hom i j)) i j) 0
                   -/
  zero i j w := by dsimp; rw [h.zero i j w, F.map_zero]
                          /-
                            🎉 no goals
                          -/
  comm i := by
    /-
      ι : Type u_1
      V : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} V
      inst✝³ : CategoryTheory.Preadditive V
      c : ComplexShape ι
      C D E : HomologicalComplex V c
      f✝ g✝ : Quiver.Hom C D
      h✝ k : Quiver.Hom D E
      i✝ : ι
      W : Type u_2
      inst✝² : CategoryTheory.Category.{?u.274677, u_2} W
      inst✝¹ : CategoryTheory.Preadditive W
      F : CategoryTheory.Functor V W
      inst✝ : F.Additive
      f g : Quiver.Hom C D
      h : Homotopy f g
      i : ι
      ⊢ Eq (((F.mapHomologicalComplex c).map f).f i) (HAdd.hAdd (HAdd.hAdd ((dNext i …
    -/
    have H := h.comm i
    /-
      ι : Type u_1
      V : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} V
      inst✝³ : CategoryTheory.Preadditive V
      c : ComplexShape ι
      C D E : HomologicalComplex V c
      f✝ g✝ : Quiver.Hom C D
      h✝ k : Quiver.Hom D E
      i✝ : ι
      W : Type u_2
      inst✝² : CategoryTheory.Category.{?u.274677, u_2} W
      inst✝¹ : CategoryTheory.Preadditive W
      F : CategoryTheory.Functor V W
      inst✝ : F.Additive
      f g : Quiver.Hom C D
      h : Homotopy f g
      i : ι
      H : Eq (f.f i) (HAdd.hAdd (HAdd.hAdd ((dNext i) h.hom) ((prevD i) h.hom)) (g.f …
      ⊢ Eq (((F.mapHomologicalComplex c).map f).f i) (HAdd.hAdd (HAdd.hAdd ((dNext i …
    -/
    dsimp [dNext, prevD] at H ⊢
    /-
      ι : Type u_1
      V : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} V
      inst✝³ : CategoryTheory.Preadditive V
      c : ComplexShape ι
      C D E : HomologicalComplex V c
      f✝ g✝ : Quiver.Hom C D
      h✝ k : Quiver.Hom D E
      i✝ : ι
      W : Type u_2
      inst✝² : CategoryTheory.Category.{?u.274677, u_2} W
      inst✝¹ : CategoryTheory.Preadditive W
      F : CategoryTheory.Functor V W
      inst✝ : F.Additive
      f g : Quiver.Hom C D
      h : Homotopy f g
      i : ι
      H : Eq (f.f i) (HAdd.hAdd (HAdd.hAdd (CategoryTheory.CategoryStruct.comp (C.d  …
      ⊢ Eq (F.map (f.f i)) (HAdd.hAdd (HAdd.hAdd (CategoryTheory.CategoryStruct.comp …
    -/
    simp [H]
    /-
      🎉 no goals
    -/


/-- An additive functor preserves homotopy equivalences. -/
@[simps]
def Functor.mapHomotopyEquiv (F : V ⥤ W) [F.Additive] (h : HomotopyEquiv C D) :
    HomotopyEquiv ((F.mapHomologicalComplex c).obj C) ((F.mapHomologicalComplex c).obj D) where
  hom := (F.mapHomologicalComplex c).map h.hom
  inv := (F.mapHomologicalComplex c).map h.inv
  homotopyHomInvId := by
    /-
      ι : Type u_1
      V : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} V
      inst✝³ : CategoryTheory.Preadditive V
      c : ComplexShape ι
      C D E : HomologicalComplex V c
      f g : Quiver.Hom C D
      h✝ k : Quiver.Hom D E
      i : ι
      W : Type u_2
      inst✝² : CategoryTheory.Category.{?u.279176, u_2} W
      inst✝¹ : CategoryTheory.Preadditive W
      F : CategoryTheory.Functor V W
      inst✝ : F.Additive
      h : HomotopyEquiv C D
      ⊢ Homotopy (CategoryTheory.CategoryStruct.comp ((F.mapHomologicalComplex c).ma …
    -/
    rw [← (F.mapHomologicalComplex c).map_comp, ← (F.mapHomologicalComplex c).map_id]
    /-
      ι : Type u_1
      V : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} V
      inst✝³ : CategoryTheory.Preadditive V
      c : ComplexShape ι
      C D E : HomologicalComplex V c
      f g : Quiver.Hom C D
      h✝ k : Quiver.Hom D E
      i : ι
      W : Type u_2
      inst✝² : CategoryTheory.Category.{?u.279176, u_2} W
      inst✝¹ : CategoryTheory.Preadditive W
      F : CategoryTheory.Functor V W
      inst✝ : F.Additive
      h : HomotopyEquiv C D
      ⊢ Homotopy ((F.mapHomologicalComplex c).map (CategoryTheory.CategoryStruct.com …
    -/
    exact F.mapHomotopy h.homotopyHomInvId
    /-
      🎉 no goals
    -/
  homotopyInvHomId := by
    /-
      ι : Type u_1
      V : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} V
      inst✝³ : CategoryTheory.Preadditive V
      c : ComplexShape ι
      C D E : HomologicalComplex V c
      f g : Quiver.Hom C D
      h✝ k : Quiver.Hom D E
      i : ι
      W : Type u_2
      inst✝² : CategoryTheory.Category.{?u.279176, u_2} W
      inst✝¹ : CategoryTheory.Preadditive W
      F : CategoryTheory.Functor V W
      inst✝ : F.Additive
      h : HomotopyEquiv C D
      ⊢ Homotopy (CategoryTheory.CategoryStruct.comp ((F.mapHomologicalComplex c).ma …
    -/
    rw [← (F.mapHomologicalComplex c).map_comp, ← (F.mapHomologicalComplex c).map_id]
    /-
      ι : Type u_1
      V : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} V
      inst✝³ : CategoryTheory.Preadditive V
      c : ComplexShape ι
      C D E : HomologicalComplex V c
      f g : Quiver.Hom C D
      h✝ k : Quiver.Hom D E
      i : ι
      W : Type u_2
      inst✝² : CategoryTheory.Category.{?u.279176, u_2} W
      inst✝¹ : CategoryTheory.Preadditive W
      F : CategoryTheory.Functor V W
      inst✝ : F.Additive
      h : HomotopyEquiv C D
      ⊢ Homotopy ((F.mapHomologicalComplex c).map (CategoryTheory.CategoryStruct.com …
    -/
    exact F.mapHomotopy h.homotopyInvHomId
    /-
      🎉 no goals
    -/


/-- A homotopy between morphisms of homological complexes `K ⟶ L` induces a homotopy
between morphisms of short complexes `K.sc i ⟶ L.sc i`. -/
noncomputable def Homotopy.toShortComplex (ho : Homotopy f g) (i : ι) :
    ShortComplex.Homotopy ((shortComplexFunctor C c i).map f)
      ((shortComplexFunctor C c i).map g) where
  h₀ :=
    if c.Rel (c.prev i) i
    then ho.hom _ (c.prev (c.prev i)) ≫ L.d _ _
    else f.f _ - g.f _ - K.d _ i ≫ ho.hom i _
  h₁ := ho.hom _ _
  h₂ := ho.hom _ _
  h₃ :=
    if c.Rel i (c.next i)
    then K.d _ _ ≫ ho.hom (c.next (c.next i)) _
    else f.f _ - g.f _ - ho.hom _ i ≫ L.d _ _
  h₀_f := by
    /-
      ι✝ : Type u_1
      V : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} V
      inst✝³ : CategoryTheory.Preadditive V
      c✝ : ComplexShape ι✝
      C✝ D E : HomologicalComplex V c✝
      f✝ g✝ : Quiver.Hom C✝ D
      h k : Quiver.Hom D E
      i✝ : ι✝
      C : Type u_2
      inst✝² : CategoryTheory.Category.{?u.282053, u_2} C
      inst✝¹ : CategoryTheory.Preadditive C
      ι : Type ?u.282072
      c : ComplexShape ι
      inst✝ : DecidableRel c.Rel
      K L : HomologicalComplex C c
      f g : Quiver.Hom K L
      ho : Homotopy f g
      i : ι
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (ite (c.Rel (c.prev i) i) (CategoryTh …
    -/
    split_ifs with h
      /-
        case pos
        ι✝ : Type u_1
        V : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} V
        inst✝³ : CategoryTheory.Preadditive V
        c✝ : ComplexShape ι✝
        C✝ D E : HomologicalComplex V c✝
        f✝ g✝ : Quiver.Hom C✝ D
        h✝ k : Quiver.Hom D E
        i✝ : ι✝
        C : Type u_2
        inst✝² : CategoryTheory.Category.{?u.282053, u_2} C
        inst✝¹ : CategoryTheory.Preadditive C
        ι : Type ?u.282072
        c : ComplexShape ι
        inst✝ : DecidableRel c.Rel
        K L : HomologicalComplex C c
        f g : Quiver.Hom K L
        ho : Homotopy f g
        i : ι
        h : c.Rel (c.prev i) i
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
    · dsimp
      /-
        case pos
        ι✝ : Type u_1
        V : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} V
        inst✝³ : CategoryTheory.Preadditive V
        c✝ : ComplexShape ι✝
        C✝ D E : HomologicalComplex V c✝
        f✝ g✝ : Quiver.Hom C✝ D
        h✝ k : Quiver.Hom D E
        i✝ : ι✝
        C : Type u_2
        inst✝² : CategoryTheory.Category.{?u.282053, u_2} C
        inst✝¹ : CategoryTheory.Preadditive C
        ι : Type ?u.282072
        c : ComplexShape ι
        inst✝ : DecidableRel c.Rel
        K L : HomologicalComplex C c
        f g : Quiver.Hom K L
        ho : Homotopy f g
        i : ι
        h : c.Rel (c.prev i) i
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
      simp only [assoc, d_comp_d, comp_zero]
      /-
        🎉 no goals
      -/
      /-
        case neg
        ι✝ : Type u_1
        V : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} V
        inst✝³ : CategoryTheory.Preadditive V
        c✝ : ComplexShape ι✝
        C✝ D E : HomologicalComplex V c✝
        f✝ g✝ : Quiver.Hom C✝ D
        h✝ k : Quiver.Hom D E
        i✝ : ι✝
        C : Type u_2
        inst✝² : CategoryTheory.Category.{?u.282053, u_2} C
        inst✝¹ : CategoryTheory.Preadditive C
        ι : Type ?u.282072
        c : ComplexShape ι
        inst✝ : DecidableRel c.Rel
        K L : HomologicalComplex C c
        f g : Quiver.Hom K L
        ho : Homotopy f g
        i : ι
        h : Not (c.Rel (c.prev i) i)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (HSub.hSub (HSub.hSub (f.f (c.prev i) …
      -/
    · dsimp
      /-
        case neg
        ι✝ : Type u_1
        V : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} V
        inst✝³ : CategoryTheory.Preadditive V
        c✝ : ComplexShape ι✝
        C✝ D E : HomologicalComplex V c✝
        f✝ g✝ : Quiver.Hom C✝ D
        h✝ k : Quiver.Hom D E
        i✝ : ι✝
        C : Type u_2
        inst✝² : CategoryTheory.Category.{?u.282053, u_2} C
        inst✝¹ : CategoryTheory.Preadditive C
        ι : Type ?u.282072
        c : ComplexShape ι
        inst✝ : DecidableRel c.Rel
        K L : HomologicalComplex C c
        f g : Quiver.Hom K L
        ho : Homotopy f g
        i : ι
        h : Not (c.Rel (c.prev i) i)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (HSub.hSub (HSub.hSub (f.f (c.prev i) …
      -/
      rw [L.shape _ _ h, comp_zero]
      /-
        🎉 no goals
      -/
  g_h₃ := by
    /-
      ι✝ : Type u_1
      V : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} V
      inst✝³ : CategoryTheory.Preadditive V
      c✝ : ComplexShape ι✝
      C✝ D E : HomologicalComplex V c✝
      f✝ g✝ : Quiver.Hom C✝ D
      h k : Quiver.Hom D E
      i✝ : ι✝
      C : Type u_2
      inst✝² : CategoryTheory.Category.{?u.282053, u_2} C
      inst✝¹ : CategoryTheory.Preadditive C
      ι : Type ?u.282072
      c : ComplexShape ι
      inst✝ : DecidableRel c.Rel
      K L : HomologicalComplex C c
      f g : Quiver.Hom K L
      ho : Homotopy f g
      i : ι
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((HomologicalComplex.shortComplexFunc …
    -/
    split_ifs with h
      /-
        case pos
        ι✝ : Type u_1
        V : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} V
        inst✝³ : CategoryTheory.Preadditive V
        c✝ : ComplexShape ι✝
        C✝ D E : HomologicalComplex V c✝
        f✝ g✝ : Quiver.Hom C✝ D
        h✝ k : Quiver.Hom D E
        i✝ : ι✝
        C : Type u_2
        inst✝² : CategoryTheory.Category.{?u.282053, u_2} C
        inst✝¹ : CategoryTheory.Preadditive C
        ι : Type ?u.282072
        c : ComplexShape ι
        inst✝ : DecidableRel c.Rel
        K L : HomologicalComplex C c
        f g : Quiver.Hom K L
        ho : Homotopy f g
        i : ι
        h : c.Rel i (c.next i)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((HomologicalComplex.shortComplexFunc …
      -/
    · dsimp
      /-
        case pos
        ι✝ : Type u_1
        V : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} V
        inst✝³ : CategoryTheory.Preadditive V
        c✝ : ComplexShape ι✝
        C✝ D E : HomologicalComplex V c✝
        f✝ g✝ : Quiver.Hom C✝ D
        h✝ k : Quiver.Hom D E
        i✝ : ι✝
        C : Type u_2
        inst✝² : CategoryTheory.Category.{?u.282053, u_2} C
        inst✝¹ : CategoryTheory.Preadditive C
        ι : Type ?u.282072
        c : ComplexShape ι
        inst✝ : DecidableRel c.Rel
        K L : HomologicalComplex C c
        f g : Quiver.Hom K L
        ho : Homotopy f g
        i : ι
        h : c.Rel i (c.next i)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.d i (c.next i)) (CategoryTheory.Ca …
      -/
      simp
      /-
        🎉 no goals
      -/
      /-
        case neg
        ι✝ : Type u_1
        V : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} V
        inst✝³ : CategoryTheory.Preadditive V
        c✝ : ComplexShape ι✝
        C✝ D E : HomologicalComplex V c✝
        f✝ g✝ : Quiver.Hom C✝ D
        h✝ k : Quiver.Hom D E
        i✝ : ι✝
        C : Type u_2
        inst✝² : CategoryTheory.Category.{?u.282053, u_2} C
        inst✝¹ : CategoryTheory.Preadditive C
        ι : Type ?u.282072
        c : ComplexShape ι
        inst✝ : DecidableRel c.Rel
        K L : HomologicalComplex C c
        f g : Quiver.Hom K L
        ho : Homotopy f g
        i : ι
        h : Not (c.Rel i (c.next i))
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((HomologicalComplex.shortComplexFunc …
      -/
    · dsimp
      /-
        case neg
        ι✝ : Type u_1
        V : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} V
        inst✝³ : CategoryTheory.Preadditive V
        c✝ : ComplexShape ι✝
        C✝ D E : HomologicalComplex V c✝
        f✝ g✝ : Quiver.Hom C✝ D
        h✝ k : Quiver.Hom D E
        i✝ : ι✝
        C : Type u_2
        inst✝² : CategoryTheory.Category.{?u.282053, u_2} C
        inst✝¹ : CategoryTheory.Preadditive C
        ι : Type ?u.282072
        c : ComplexShape ι
        inst✝ : DecidableRel c.Rel
        K L : HomologicalComplex C c
        f g : Quiver.Hom K L
        ho : Homotopy f g
        i : ι
        h : Not (c.Rel i (c.next i))
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.d i (c.next i)) (HSub.hSub (HSub.h …
      -/
      rw [K.shape _ _ h, zero_comp]
      /-
        🎉 no goals
      -/
  comm₁ := by
    /-
      ι✝ : Type u_1
      V : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} V
      inst✝³ : CategoryTheory.Preadditive V
      c✝ : ComplexShape ι✝
      C✝ D E : HomologicalComplex V c✝
      f✝ g✝ : Quiver.Hom C✝ D
      h k : Quiver.Hom D E
      i✝ : ι✝
      C : Type u_2
      inst✝² : CategoryTheory.Category.{?u.282053, u_2} C
      inst✝¹ : CategoryTheory.Preadditive C
      ι : Type ?u.282072
      c : ComplexShape ι
      inst✝ : DecidableRel c.Rel
      K L : HomologicalComplex C c
      f g : Quiver.Hom K L
      ho : Homotopy f g
      i : ι
      ⊢ Eq ((HomologicalComplex.shortComplexFunctor C c i).map f).τ₁ (HAdd.hAdd (HAd …
    -/
    dsimp
    /-
      ι✝ : Type u_1
      V : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} V
      inst✝³ : CategoryTheory.Preadditive V
      c✝ : ComplexShape ι✝
      C✝ D E : HomologicalComplex V c✝
      f✝ g✝ : Quiver.Hom C✝ D
      h k : Quiver.Hom D E
      i✝ : ι✝
      C : Type u_2
      inst✝² : CategoryTheory.Category.{?u.282053, u_2} C
      inst✝¹ : CategoryTheory.Preadditive C
      ι : Type ?u.282072
      c : ComplexShape ι
      inst✝ : DecidableRel c.Rel
      K L : HomologicalComplex C c
      f g : Quiver.Hom K L
      ho : Homotopy f g
      i : ι
      ⊢ Eq (f.f (c.prev i)) (HAdd.hAdd (HAdd.hAdd (CategoryTheory.CategoryStruct.com …
    -/
    split_ifs with h
      /-
        case pos
        ι✝ : Type u_1
        V : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} V
        inst✝³ : CategoryTheory.Preadditive V
        c✝ : ComplexShape ι✝
        C✝ D E : HomologicalComplex V c✝
        f✝ g✝ : Quiver.Hom C✝ D
        h✝ k : Quiver.Hom D E
        i✝ : ι✝
        C : Type u_2
        inst✝² : CategoryTheory.Category.{?u.282053, u_2} C
        inst✝¹ : CategoryTheory.Preadditive C
        ι : Type ?u.282072
        c : ComplexShape ι
        inst✝ : DecidableRel c.Rel
        K L : HomologicalComplex C c
        f g : Quiver.Hom K L
        ho : Homotopy f g
        i : ι
        h : c.Rel (c.prev i) i
        ⊢ Eq (f.f (c.prev i)) (HAdd.hAdd (HAdd.hAdd (CategoryTheory.CategoryStruct.com …
      -/
    · rw [ho.comm (c.prev i)]
      /-
        case pos
        ι✝ : Type u_1
        V : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} V
        inst✝³ : CategoryTheory.Preadditive V
        c✝ : ComplexShape ι✝
        C✝ D E : HomologicalComplex V c✝
        f✝ g✝ : Quiver.Hom C✝ D
        h✝ k : Quiver.Hom D E
        i✝ : ι✝
        C : Type u_2
        inst✝² : CategoryTheory.Category.{?u.282053, u_2} C
        inst✝¹ : CategoryTheory.Preadditive C
        ι : Type ?u.282072
        c : ComplexShape ι
        inst✝ : DecidableRel c.Rel
        K L : HomologicalComplex C c
        f g : Quiver.Hom K L
        ho : Homotopy f g
        i : ι
        h : c.Rel (c.prev i) i
        ⊢ Eq (HAdd.hAdd (HAdd.hAdd ((dNext (c.prev i)) ho.hom) ((prevD (c.prev i)) ho. …
      -/
      dsimp [dFrom, dTo, fromNext, toPrev]
      /-
        case pos
        ι✝ : Type u_1
        V : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} V
        inst✝³ : CategoryTheory.Preadditive V
        c✝ : ComplexShape ι✝
        C✝ D E : HomologicalComplex V c✝
        f✝ g✝ : Quiver.Hom C✝ D
        h✝ k : Quiver.Hom D E
        i✝ : ι✝
        C : Type u_2
        inst✝² : CategoryTheory.Category.{?u.282053, u_2} C
        inst✝¹ : CategoryTheory.Preadditive C
        ι : Type ?u.282072
        c : ComplexShape ι
        inst✝ : DecidableRel c.Rel
        K L : HomologicalComplex C c
        f g : Quiver.Hom K L
        ho : Homotopy f g
        i : ι
        h : c.Rel (c.prev i) i
        ⊢ Eq (HAdd.hAdd (HAdd.hAdd (CategoryTheory.CategoryStruct.comp (K.d (c.prev i) …
      -/
      rw [congr_arg (fun j => d K (c.prev i) j ≫ ho.hom j (c.prev i)) (c.next_eq' h)]
      /-
        🎉 no goals
      -/
      /-
        case neg
        ι✝ : Type u_1
        V : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} V
        inst✝³ : CategoryTheory.Preadditive V
        c✝ : ComplexShape ι✝
        C✝ D E : HomologicalComplex V c✝
        f✝ g✝ : Quiver.Hom C✝ D
        h✝ k : Quiver.Hom D E
        i✝ : ι✝
        C : Type u_2
        inst✝² : CategoryTheory.Category.{?u.282053, u_2} C
        inst✝¹ : CategoryTheory.Preadditive C
        ι : Type ?u.282072
        c : ComplexShape ι
        inst✝ : DecidableRel c.Rel
        K L : HomologicalComplex C c
        f g : Quiver.Hom K L
        ho : Homotopy f g
        i : ι
        h : Not (c.Rel (c.prev i) i)
        ⊢ Eq (f.f (c.prev i)) (HAdd.hAdd (HAdd.hAdd (CategoryTheory.CategoryStruct.com …
      -/
      /-
        🎉 no goals
      -/
    · abel
      /-
        🎉 no goals
      -/
  comm₂ := ho.comm i
  comm₃ := by
    /-
      ι✝ : Type u_1
      V : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} V
      inst✝³ : CategoryTheory.Preadditive V
      c✝ : ComplexShape ι✝
      C✝ D E : HomologicalComplex V c✝
      f✝ g✝ : Quiver.Hom C✝ D
      h k : Quiver.Hom D E
      i✝ : ι✝
      C : Type u_2
      inst✝² : CategoryTheory.Category.{?u.282053, u_2} C
      inst✝¹ : CategoryTheory.Preadditive C
      ι : Type ?u.282072
      c : ComplexShape ι
      inst✝ : DecidableRel c.Rel
      K L : HomologicalComplex C c
      f g : Quiver.Hom K L
      ho : Homotopy f g
      i : ι
      ⊢ Eq ((HomologicalComplex.shortComplexFunctor C c i).map f).τ₃ (HAdd.hAdd (HAd …
    -/
    dsimp
    /-
      ι✝ : Type u_1
      V : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} V
      inst✝³ : CategoryTheory.Preadditive V
      c✝ : ComplexShape ι✝
      C✝ D E : HomologicalComplex V c✝
      f✝ g✝ : Quiver.Hom C✝ D
      h k : Quiver.Hom D E
      i✝ : ι✝
      C : Type u_2
      inst✝² : CategoryTheory.Category.{?u.282053, u_2} C
      inst✝¹ : CategoryTheory.Preadditive C
      ι : Type ?u.282072
      c : ComplexShape ι
      inst✝ : DecidableRel c.Rel
      K L : HomologicalComplex C c
      f g : Quiver.Hom K L
      ho : Homotopy f g
      i : ι
      ⊢ Eq (f.f (c.next i)) (HAdd.hAdd (HAdd.hAdd (ite (c.Rel i (c.next i)) (Categor …
    -/
    split_ifs with h
      /-
        case pos
        ι✝ : Type u_1
        V : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} V
        inst✝³ : CategoryTheory.Preadditive V
        c✝ : ComplexShape ι✝
        C✝ D E : HomologicalComplex V c✝
        f✝ g✝ : Quiver.Hom C✝ D
        h✝ k : Quiver.Hom D E
        i✝ : ι✝
        C : Type u_2
        inst✝² : CategoryTheory.Category.{?u.282053, u_2} C
        inst✝¹ : CategoryTheory.Preadditive C
        ι : Type ?u.282072
        c : ComplexShape ι
        inst✝ : DecidableRel c.Rel
        K L : HomologicalComplex C c
        f g : Quiver.Hom K L
        ho : Homotopy f g
        i : ι
        h : c.Rel i (c.next i)
        ⊢ Eq (f.f (c.next i)) (HAdd.hAdd (HAdd.hAdd (CategoryTheory.CategoryStruct.com …
      -/
    · rw [ho.comm (c.next i)]
      /-
        case pos
        ι✝ : Type u_1
        V : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} V
        inst✝³ : CategoryTheory.Preadditive V
        c✝ : ComplexShape ι✝
        C✝ D E : HomologicalComplex V c✝
        f✝ g✝ : Quiver.Hom C✝ D
        h✝ k : Quiver.Hom D E
        i✝ : ι✝
        C : Type u_2
        inst✝² : CategoryTheory.Category.{?u.282053, u_2} C
        inst✝¹ : CategoryTheory.Preadditive C
        ι : Type ?u.282072
        c : ComplexShape ι
        inst✝ : DecidableRel c.Rel
        K L : HomologicalComplex C c
        f g : Quiver.Hom K L
        ho : Homotopy f g
        i : ι
        h : c.Rel i (c.next i)
        ⊢ Eq (HAdd.hAdd (HAdd.hAdd ((dNext (c.next i)) ho.hom) ((prevD (c.next i)) ho. …
      -/
      dsimp [dFrom, dTo, fromNext, toPrev]
      /-
        case pos
        ι✝ : Type u_1
        V : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} V
        inst✝³ : CategoryTheory.Preadditive V
        c✝ : ComplexShape ι✝
        C✝ D E : HomologicalComplex V c✝
        f✝ g✝ : Quiver.Hom C✝ D
        h✝ k : Quiver.Hom D E
        i✝ : ι✝
        C : Type u_2
        inst✝² : CategoryTheory.Category.{?u.282053, u_2} C
        inst✝¹ : CategoryTheory.Preadditive C
        ι : Type ?u.282072
        c : ComplexShape ι
        inst✝ : DecidableRel c.Rel
        K L : HomologicalComplex C c
        f g : Quiver.Hom K L
        ho : Homotopy f g
        i : ι
        h : c.Rel i (c.next i)
        ⊢ Eq (HAdd.hAdd (HAdd.hAdd (CategoryTheory.CategoryStruct.comp (K.d (c.next i) …
      -/
      rw [congr_arg (fun j => ho.hom (c.next i) j ≫ L.d j (c.next i)) (c.prev_eq' h)]
      /-
        🎉 no goals
      -/
      /-
        case neg
        ι✝ : Type u_1
        V : Type u
        inst✝⁴ : CategoryTheory.Category.{v, u} V
        inst✝³ : CategoryTheory.Preadditive V
        c✝ : ComplexShape ι✝
        C✝ D E : HomologicalComplex V c✝
        f✝ g✝ : Quiver.Hom C✝ D
        h✝ k : Quiver.Hom D E
        i✝ : ι✝
        C : Type u_2
        inst✝² : CategoryTheory.Category.{?u.282053, u_2} C
        inst✝¹ : CategoryTheory.Preadditive C
        ι : Type ?u.282072
        c : ComplexShape ι
        inst✝ : DecidableRel c.Rel
        K L : HomologicalComplex C c
        f g : Quiver.Hom K L
        ho : Homotopy f g
        i : ι
        h : Not (c.Rel i (c.next i))
        ⊢ Eq (f.f (c.next i)) (HAdd.hAdd (HAdd.hAdd (HSub.hSub (HSub.hSub (f.f (c.next …
      -/
      /-
        🎉 no goals
      -/
    · abel
      /-
        🎉 no goals
      -/


lemma Homotopy.homologyMap_eq (ho : Homotopy f g) (i : ι) [K.HasHomology i] [L.HasHomology i] :
    homologyMap f i = homologyMap g i :=
  ShortComplex.Homotopy.homologyMap_congr (ho.toShortComplex i)


/-- The isomorphism in homology induced by an homotopy equivalence. -/
noncomputable def HomotopyEquiv.toHomologyIso (h : HomotopyEquiv K L) (i : ι)
    [K.HasHomology i] [L.HasHomology i] : K.homology i ≅ L.homology i where
  hom := homologyMap h.hom i
  inv := homologyMap h.inv i
                   /-
                     ι✝ : Type u_1
                     V : Type u
                     inst✝⁶ : CategoryTheory.Category.{v, u} V
                     inst✝⁵ : CategoryTheory.Preadditive V
                     c✝ : ComplexShape ι✝
                     C✝ D E : HomologicalComplex V c✝
                     f✝ g✝ : Quiver.Hom C✝ D
                     h✝ k : Quiver.Hom D E
                     i✝ : ι✝
                     C : Type u_2
                     inst✝⁴ : CategoryTheory.Category.{?u.297276, u_2} C
                     inst✝³ : CategoryTheory.Preadditive C
                     ι : Type ?u.297295
                     c : ComplexShape ι
                     inst✝² : DecidableRel c.Rel
                     K L : HomologicalComplex C c
                     f g : Quiver.Hom K L
                     h : HomotopyEquiv K L
                     i : ι
                     inst✝¹ : K.HasHomology i
                     inst✝ : L.HasHomology i
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.homologyMap h.hom …
                   -/
  hom_inv_id := by rw [← homologyMap_comp, h.homotopyHomInvId.homologyMap_eq, homologyMap_id]
                   /-
                     🎉 no goals
                   -/
                   /-
                     ι✝ : Type u_1
                     V : Type u
                     inst✝⁶ : CategoryTheory.Category.{v, u} V
                     inst✝⁵ : CategoryTheory.Preadditive V
                     c✝ : ComplexShape ι✝
                     C✝ D E : HomologicalComplex V c✝
                     f✝ g✝ : Quiver.Hom C✝ D
                     h✝ k : Quiver.Hom D E
                     i✝ : ι✝
                     C : Type u_2
                     inst✝⁴ : CategoryTheory.Category.{?u.297276, u_2} C
                     inst✝³ : CategoryTheory.Preadditive C
                     ι : Type ?u.297295
                     c : ComplexShape ι
                     inst✝² : DecidableRel c.Rel
                     K L : HomologicalComplex C c
                     f g : Quiver.Hom K L
                     h : HomotopyEquiv K L
                     i : ι
                     inst✝¹ : K.HasHomology i
                     inst✝ : L.HasHomology i
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.homologyMap h.inv …
                   -/
  inv_hom_id := by rw [← homologyMap_comp, h.homotopyInvHomId.homologyMap_eq, homologyMap_id]
                   /-
                     🎉 no goals
                   -/


