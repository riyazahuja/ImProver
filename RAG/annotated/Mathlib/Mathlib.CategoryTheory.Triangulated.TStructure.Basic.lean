/-- `TStructure C` is the type of t-structures on the (pre)triangulated category `C`. -/
structure TStructure where
  /-- the predicate of objects that are `≤ n` for `n : ℤ`. -/
  LE (n : ℤ) : C → Prop
  /-- the predicate of objects that are `≥ n` for `n : ℤ`. -/
  GE (n : ℤ) : C → Prop
  LE_closedUnderIsomorphisms (n : ℤ) : ClosedUnderIsomorphisms (LE n) := by infer_instance
  GE_closedUnderIsomorphisms (n : ℤ) : ClosedUnderIsomorphisms (GE n) := by infer_instance
  LE_shift (n a n' : ℤ) (h : a + n' = n) (X : C) (hX : LE n X) : LE n' (X⟦a⟧)
  GE_shift (n a n' : ℤ) (h : a + n' = n) (X : C) (hX : GE n X) : GE n' (X⟦a⟧)
  zero' ⦃X Y : C⦄ (f : X ⟶ Y) (hX : LE 0 X) (hY : GE 1 Y) : f = 0
  LE_zero_le : LE 0 ≤ LE 1
  GE_one_le : GE 1 ≤ GE 0
  exists_triangle_zero_one (A : C) : ∃ (X Y : C) (_ : LE 0 X) (_ : GE 1 Y)
    (f : X ⟶ A) (g : A ⟶ Y) (h : Y ⟶ X⟦(1 : ℤ)⟧), Triangle.mk f g h ∈ distTriang C


lemma exists_triangle (A : C) (n₀ n₁ : ℤ) (h : n₀ + 1 = n₁) :
    ∃ (X Y : C) (_ : t.LE n₀ X) (_ : t.GE n₁ Y) (f : X ⟶ A) (g : A ⟶ Y)
      (h : Y ⟶ X⟦(1 : ℤ)⟧), Triangle.mk f g h ∈ distTriang C := by
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁴ : CategoryTheory.Preadditive C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝ : CategoryTheory.Pretriangulated C
    t : CategoryTheory.Triangulated.TStructure C
    A : C
    n₀ n₁ : Int
    h : Eq (HAdd.hAdd n₀ 1) n₁
    ⊢ Exists fun X => Exists fun Y => Exists fun x => Exists fun x => Exists fun f …
  -/
  obtain ⟨X, Y, hX, hY, f, g, h, mem⟩ := t.exists_triangle_zero_one (A⟦n₀⟧)
  /-
    case intro.intro.intro.intro.intro.intro.intro
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁴ : CategoryTheory.Preadditive C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝ : CategoryTheory.Pretriangulated C
    t : CategoryTheory.Triangulated.TStructure C
    A : C
    n₀ n₁ : Int
    h✝ : Eq (HAdd.hAdd n₀ 1) n₁
    X Y : C
    hX : t.LE 0 X
    hY : t.GE 1 Y
    f : Quiver.Hom X ((CategoryTheory.shiftFunctor C n₀).obj A)
    g : Quiver.Hom ((CategoryTheory.shiftFunctor C n₀).obj A) Y
    h : Quiver.Hom Y ((CategoryTheory.shiftFunctor C 1).obj X)
    mem : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
    ⊢ Exists fun X => Exists fun Y => Exists fun x => Exists fun x => Exists fun f …
  -/
  let T := (Triangle.shiftFunctor C (-n₀)).obj (Triangle.mk f g h)
  /-
    case intro.intro.intro.intro.intro.intro.intro
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁴ : CategoryTheory.Preadditive C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝ : CategoryTheory.Pretriangulated C
    t : CategoryTheory.Triangulated.TStructure C
    A : C
    n₀ n₁ : Int
    h✝ : Eq (HAdd.hAdd n₀ 1) n₁
    X Y : C
    hX : t.LE 0 X
    hY : t.GE 1 Y
    f : Quiver.Hom X ((CategoryTheory.shiftFunctor C n₀).obj A)
    g : Quiver.Hom ((CategoryTheory.shiftFunctor C n₀).obj A) Y
    h : Quiver.Hom Y ((CategoryTheory.shiftFunctor C 1).obj X)
    mem : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
    T : CategoryTheory.Pretriangulated.Triangle C := (CategoryTheory.Pretriangulat …
    ⊢ Exists fun X => Exists fun Y => Exists fun x => Exists fun x => Exists fun f …
  -/
  let e := (shiftEquiv C n₀).unitIso.symm.app A
  have hT' : Triangle.mk (T.mor₁ ≫ e.hom) (e.inv ≫ T.mor₂) T.mor₃ ∈ distTriang C := by
    refine isomorphic_distinguished _ (Triangle.shift_distinguished _ mem (-n₀)) _ ?_
    refine Triangle.isoMk _ _ (Iso.refl _) e.symm (Iso.refl _) ?_ ?_ ?_
    all_goals dsimp; simp [T]
  exact ⟨_, _, t.LE_shift _ _ _ (neg_add_cancel n₀) _ hX,
    t.GE_shift _ _ _ (by omega) _ hY, _, _, _, hT'⟩


lemma predicateShift_LE (a n n' : ℤ) (hn' : a + n = n') :
    (PredicateShift (t.LE n) a) = t.LE n' := by
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁴ : CategoryTheory.Preadditive C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝ : CategoryTheory.Pretriangulated C
    t : CategoryTheory.Triangulated.TStructure C
    a n n' : Int
    hn' : Eq (HAdd.hAdd a n) n'
    ⊢ Eq (CategoryTheory.PredicateShift (t.LE n) a) (t.LE n')
  -/
  ext X
  /-
    case h.a
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁴ : CategoryTheory.Preadditive C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝ : CategoryTheory.Pretriangulated C
    t : CategoryTheory.Triangulated.TStructure C
    a n n' : Int
    hn' : Eq (HAdd.hAdd a n) n'
    X : C
    ⊢ Iff (CategoryTheory.PredicateShift (t.LE n) a X) (t.LE n' X)
  -/
  constructor
    /-
      case h.a.mp
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
      inst✝⁴ : CategoryTheory.Preadditive C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝ : CategoryTheory.Pretriangulated C
      t : CategoryTheory.Triangulated.TStructure C
      a n n' : Int
      hn' : Eq (HAdd.hAdd a n) n'
      X : C
      ⊢ CategoryTheory.PredicateShift (t.LE n) a X → t.LE n' X
    -/
  · intro hX
    exact (mem_iff_of_iso (LE t n') ((shiftEquiv C a).unitIso.symm.app X)).1
      (t.LE_shift n (-a) n' (by omega) _ hX)
    /-
      case h.a.mpr
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
      inst✝⁴ : CategoryTheory.Preadditive C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝ : CategoryTheory.Pretriangulated C
      t : CategoryTheory.Triangulated.TStructure C
      a n n' : Int
      hn' : Eq (HAdd.hAdd a n) n'
      X : C
      ⊢ t.LE n' X → CategoryTheory.PredicateShift (t.LE n) a X
    -/
  · intro hX
    /-
      case h.a.mpr
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
      inst✝⁴ : CategoryTheory.Preadditive C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝ : CategoryTheory.Pretriangulated C
      t : CategoryTheory.Triangulated.TStructure C
      a n n' : Int
      hn' : Eq (HAdd.hAdd a n) n'
      X : C
      hX : t.LE n' X
      ⊢ CategoryTheory.PredicateShift (t.LE n) a X
    -/
    exact t.LE_shift _ _ _ hn' X hX
    /-
      🎉 no goals
    -/


lemma predicateShift_GE (a n n' : ℤ) (hn' : a + n = n') :
    (PredicateShift (t.GE n) a) = t.GE n' := by
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁴ : CategoryTheory.Preadditive C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝ : CategoryTheory.Pretriangulated C
    t : CategoryTheory.Triangulated.TStructure C
    a n n' : Int
    hn' : Eq (HAdd.hAdd a n) n'
    ⊢ Eq (CategoryTheory.PredicateShift (t.GE n) a) (t.GE n')
  -/
  ext X
  /-
    case h.a
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁴ : CategoryTheory.Preadditive C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝ : CategoryTheory.Pretriangulated C
    t : CategoryTheory.Triangulated.TStructure C
    a n n' : Int
    hn' : Eq (HAdd.hAdd a n) n'
    X : C
    ⊢ Iff (CategoryTheory.PredicateShift (t.GE n) a X) (t.GE n' X)
  -/
  constructor
    /-
      case h.a.mp
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
      inst✝⁴ : CategoryTheory.Preadditive C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝ : CategoryTheory.Pretriangulated C
      t : CategoryTheory.Triangulated.TStructure C
      a n n' : Int
      hn' : Eq (HAdd.hAdd a n) n'
      X : C
      ⊢ CategoryTheory.PredicateShift (t.GE n) a X → t.GE n' X
    -/
  · intro hX
    exact (mem_iff_of_iso (GE t n') ((shiftEquiv C a).unitIso.symm.app X)).1
      (t.GE_shift n (-a) n' (by omega) _ hX)
    /-
      case h.a.mpr
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
      inst✝⁴ : CategoryTheory.Preadditive C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝ : CategoryTheory.Pretriangulated C
      t : CategoryTheory.Triangulated.TStructure C
      a n n' : Int
      hn' : Eq (HAdd.hAdd a n) n'
      X : C
      ⊢ t.GE n' X → CategoryTheory.PredicateShift (t.GE n) a X
    -/
  · intro hX
    /-
      case h.a.mpr
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
      inst✝⁴ : CategoryTheory.Preadditive C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝ : CategoryTheory.Pretriangulated C
      t : CategoryTheory.Triangulated.TStructure C
      a n n' : Int
      hn' : Eq (HAdd.hAdd a n) n'
      X : C
      hX : t.GE n' X
      ⊢ CategoryTheory.PredicateShift (t.GE n) a X
    -/
    exact t.GE_shift _ _ _ hn' X hX
    /-
      🎉 no goals
    -/


lemma LE_monotone : Monotone t.LE := by
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁴ : CategoryTheory.Preadditive C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝ : CategoryTheory.Pretriangulated C
    t : CategoryTheory.Triangulated.TStructure C
    ⊢ Monotone t.LE
  -/
  let H := fun (a : ℕ) => ∀ (n : ℤ), t.LE n ≤ t.LE (n + a)
  suffices ∀ (a : ℕ), H a by
    intro n₀ n₁ h
    obtain ⟨a, ha⟩ := Int.nonneg_def.1 h
    obtain rfl : n₁ = n₀ + a := by omega
    apply this
  have H_zero : H 0 := fun n => by
    simp only [Nat.cast_zero, add_zero]
    rfl
  have H_one : H 1 := fun n X hX => by
    rw [← t.predicateShift_LE n 1 (n + (1 : ℕ)) rfl, predicateShift_iff]
    rw [← t.predicateShift_LE n 0 n (add_zero n), predicateShift_iff] at hX
    exact t.LE_zero_le _ hX
  have H_add : ∀ (a b c : ℕ) (_ : a + b = c) (_ : H a) (_ : H b), H c := by
    intro a b c h ha hb n
    rw [← h, Nat.cast_add, ← add_assoc]
    exact (ha n).trans (hb (n+a))
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁴ : CategoryTheory.Preadditive C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝ : CategoryTheory.Pretriangulated C
    t : CategoryTheory.Triangulated.TStructure C
    H : Nat → Prop := fun a => ∀ (n : Int), LE.le (t.LE n) (t.LE (HAdd.hAdd n ↑a))
    H_zero : H 0
    H_one : H 1
    H_add : ∀ (a b c : Nat), Eq (HAdd.hAdd a b) c → H a → H b → H c
    ⊢ ∀ (a : Nat), H a
  -/
  intro a
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁴ : CategoryTheory.Preadditive C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝ : CategoryTheory.Pretriangulated C
    t : CategoryTheory.Triangulated.TStructure C
    H : Nat → Prop := fun a => ∀ (n : Int), LE.le (t.LE n) (t.LE (HAdd.hAdd n ↑a))
    H_zero : H 0
    H_one : H 1
    H_add : ∀ (a b c : Nat), Eq (HAdd.hAdd a b) c → H a → H b → H c
    a : Nat
    ⊢ H a
  -/
  induction' a with a ha
    /-
      case zero
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
      inst✝⁴ : CategoryTheory.Preadditive C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝ : CategoryTheory.Pretriangulated C
      t : CategoryTheory.Triangulated.TStructure C
      H : Nat → Prop := fun a => ∀ (n : Int), LE.le (t.LE n) (t.LE (HAdd.hAdd n ↑a))
      H_zero : H 0
      H_one : H 1
      H_add : ∀ (a b c : Nat), Eq (HAdd.hAdd a b) c → H a → H b → H c
      ⊢ H 0
    -/
  · exact H_zero
    /-
      🎉 no goals
    -/
    /-
      case succ
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
      inst✝⁴ : CategoryTheory.Preadditive C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝ : CategoryTheory.Pretriangulated C
      t : CategoryTheory.Triangulated.TStructure C
      H : Nat → Prop := fun a => ∀ (n : Int), LE.le (t.LE n) (t.LE (HAdd.hAdd n ↑a))
      H_zero : H 0
      H_one : H 1
      H_add : ∀ (a b c : Nat), Eq (HAdd.hAdd a b) c → H a → H b → H c
      a : Nat
      ha : H a
      ⊢ H (HAdd.hAdd a 1)
    -/
  · exact H_add a 1 _ rfl ha H_one
    /-
      🎉 no goals
    -/


lemma GE_antitone : Antitone t.GE := by
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁴ : CategoryTheory.Preadditive C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝ : CategoryTheory.Pretriangulated C
    t : CategoryTheory.Triangulated.TStructure C
    ⊢ Antitone t.GE
  -/
  let H := fun (a : ℕ) => ∀ (n : ℤ), t.GE (n + a) ≤ t.GE n
  suffices ∀ (a : ℕ), H a by
    intro n₀ n₁ h
    obtain ⟨a, ha⟩ := Int.nonneg_def.1 h
    obtain rfl : n₁ = n₀ + a := by omega
    apply this
  have H_zero : H 0 := fun n => by
    simp only [Nat.cast_zero, add_zero]
    rfl
  have H_one : H 1 := fun n X hX => by
    rw [← t.predicateShift_GE n 1 (n + (1 : ℕ)) (by simp), predicateShift_iff] at hX
    rw [← t.predicateShift_GE n 0 n (add_zero n)]
    exact t.GE_one_le _ hX
  have H_add : ∀ (a b c : ℕ) (_ : a + b = c) (_ : H a) (_ : H b), H c := by
    intro a b c h ha hb n
    rw [← h, Nat.cast_add, ← add_assoc ]
    exact (hb (n + a)).trans (ha n)
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁴ : CategoryTheory.Preadditive C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝ : CategoryTheory.Pretriangulated C
    t : CategoryTheory.Triangulated.TStructure C
    H : Nat → Prop := fun a => ∀ (n : Int), LE.le (t.GE (HAdd.hAdd n ↑a)) (t.GE n)
    H_zero : H 0
    H_one : H 1
    H_add : ∀ (a b c : Nat), Eq (HAdd.hAdd a b) c → H a → H b → H c
    ⊢ ∀ (a : Nat), H a
  -/
  intro a
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁴ : CategoryTheory.Preadditive C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝ : CategoryTheory.Pretriangulated C
    t : CategoryTheory.Triangulated.TStructure C
    H : Nat → Prop := fun a => ∀ (n : Int), LE.le (t.GE (HAdd.hAdd n ↑a)) (t.GE n)
    H_zero : H 0
    H_one : H 1
    H_add : ∀ (a b c : Nat), Eq (HAdd.hAdd a b) c → H a → H b → H c
    a : Nat
    ⊢ H a
  -/
  induction' a with a ha
    /-
      case zero
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
      inst✝⁴ : CategoryTheory.Preadditive C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝ : CategoryTheory.Pretriangulated C
      t : CategoryTheory.Triangulated.TStructure C
      H : Nat → Prop := fun a => ∀ (n : Int), LE.le (t.GE (HAdd.hAdd n ↑a)) (t.GE n)
      H_zero : H 0
      H_one : H 1
      H_add : ∀ (a b c : Nat), Eq (HAdd.hAdd a b) c → H a → H b → H c
      ⊢ H 0
    -/
  · exact H_zero
    /-
      🎉 no goals
    -/
    /-
      case succ
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
      inst✝⁴ : CategoryTheory.Preadditive C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝ : CategoryTheory.Pretriangulated C
      t : CategoryTheory.Triangulated.TStructure C
      H : Nat → Prop := fun a => ∀ (n : Int), LE.le (t.GE (HAdd.hAdd n ↑a)) (t.GE n)
      H_zero : H 0
      H_one : H 1
      H_add : ∀ (a b c : Nat), Eq (HAdd.hAdd a b) c → H a → H b → H c
      a : Nat
      ha : H a
      ⊢ H (HAdd.hAdd a 1)
    -/
  · exact H_add a 1 _ rfl ha H_one
    /-
      🎉 no goals
    -/


/-- Given a t-structure `t` on a pretriangulated category `C`, the property `t.IsLE X n`
holds if `X : C` is `≤ n` for the t-structure. -/
class IsLE (X : C) (n : ℤ) : Prop where
  le : t.LE n X


/-- Given a t-structure `t` on a pretriangulated category `C`, the property `t.IsGE X n`
holds if `X : C` is `≥ n` for the t-structure. -/
class IsGE (X : C) (n : ℤ) : Prop where
  ge : t.GE n X


lemma mem_of_isLE (X : C) (n : ℤ) [t.IsLE X n] : t.LE n X := IsLE.le


lemma mem_of_isGE (X : C) (n : ℤ) [t.IsGE X n] : t.GE n X := IsGE.ge


