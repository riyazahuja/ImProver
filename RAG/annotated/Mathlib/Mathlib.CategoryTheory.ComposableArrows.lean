/-- `ComposableArrows C n` is the type of functors `Fin (n + 1) ⥤ C`. -/
abbrev ComposableArrows (n : ℕ) := Fin (n + 1) ⥤ C


/-- A wrapper for `omega` which prefaces it with some quick and useful attempts -/
macro "valid" : tactic =>
  `(tactic| first | assumption | apply zero_le | apply le_rfl | transitivity <;> assumption | omega)


/-- The `i`th object (with `i : ℕ` such that `i ≤ n`) of `F : ComposableArrows C n`. -/
@[simp]
                                                                 /-
                                                                   C : Type u_1
                                                                   inst✝ : CategoryTheory.Category.{?u.878, u_1} C
                                                                   n m : Nat
                                                                   F G : CategoryTheory.ComposableArrows C n
                                                                   i : Nat
                                                                   hi : autoParam (LE.le i n) _auto✝
                                                                   ⊢ LT.lt i (HAdd.hAdd n 1)
                                                                 -/
abbrev obj' (i : ℕ) (hi : i ≤ n := by valid) : C := F.obj ⟨i, by omega⟩
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


/-- The map `F.obj' i ⟶ F.obj' j` when `F : ComposableArrows C n`, and `i` and `j`
are natural numbers such that `i ≤ j ≤ n`. -/
@[simp]
abbrev map' (i j : ℕ) (hij : i ≤ j := by valid) (hjn : j ≤ n := by valid) :
               /-
                 C : Type u_1
                 inst✝ : CategoryTheory.Category.{?u.1118, u_1} C
                 n m : Nat
                 F G : CategoryTheory.ComposableArrows C n
                 i j : Nat
                 hij : autoParam (LE.le i j) _auto✝
                 hjn : autoParam (LE.le j n) _auto✝
                 ⊢ LT.lt i (HAdd.hAdd n 1)
               -/
               /-
                 🎉 no goals
               -/
  F.obj ⟨i, by omega⟩ ⟶ F.obj ⟨j, by omega⟩ := F.map (homOfLE (by
                                     /-
                                       🎉 no goals
                                     -/
    /-
      C : Type u_1
      inst✝ : CategoryTheory.Category.{?u.1118, u_1} C
      n m : Nat
      F G : CategoryTheory.ComposableArrows C n
      i j : Nat
      hij : autoParam (LE.le i j) _auto✝
      hjn : autoParam (LE.le j n) _auto✝
      ⊢ LE.le ⟨i, ⋯⟩ ⟨j, ⋯⟩
    -/
    simp only [Fin.mk_le_mk]
    /-
      C : Type u_1
      inst✝ : CategoryTheory.Category.{?u.1118, u_1} C
      n m : Nat
      F G : CategoryTheory.ComposableArrows C n
      i j : Nat
      hij : autoParam (LE.le i j) _auto✝
      hjn : autoParam (LE.le j n) _auto✝
      ⊢ LE.le i j
    -/
    valid))
    /-
      🎉 no goals
    -/


lemma map'_self (i : ℕ) (hi : i ≤ n := by valid) :
    /-
      C : Type u_1
      inst✝ : CategoryTheory.Category.{?u.1659, u_1} C
      n m : Nat
      F G : CategoryTheory.ComposableArrows C n
      i : Nat
      hi : autoParam (LE.le i n) _auto✝
      ⊢ LE.le i i
    -/
    /-
      🎉 no goals
    -/
    F.map' i i = 𝟙 _ := F.map_id _
    /-
      🎉 no goals
    -/


lemma map'_comp (i j k : ℕ) (hij : i ≤ j := by valid)
    (hjk : j ≤ k := by valid) (hk : k ≤ n := by valid) :
    /-
      C : Type u_1
      inst✝ : CategoryTheory.Category.{?u.1936, u_1} C
      n m : Nat
      F G : CategoryTheory.ComposableArrows C n
      i j k : Nat
      hij : autoParam (LE.le i j) _auto✝
      hjk : autoParam (LE.le j k) _auto✝
      hk : autoParam (LE.le k n) _auto✝
      ⊢ LE.le i k
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
                 /-
                   🎉 no goals
                 -/
                 /-
                   🎉 no goals
                 -/
                              /-
                                🎉 no goals
                              -/
    F.map' i k = F.map' i j ≫ F.map' j k :=
                              /-
                                🎉 no goals
                              -/
  F.map_comp _ _


/-- The leftmost object of `F : ComposableArrows C n`. -/
               /-
                 C : Type u_1
                 inst✝ : CategoryTheory.Category.{?u.2438, u_1} C
                 n m : Nat
                 F G : CategoryTheory.ComposableArrows C n
                 ⊢ LE.le 0 n
               -/
abbrev left := obj' F 0
               /-
                 🎉 no goals
               -/


/-- The rightmost object of `F : ComposableArrows C n`. -/
                /-
                  C : Type u_1
                  inst✝ : CategoryTheory.Category.{?u.2753, u_1} C
                  n m : Nat
                  F G : CategoryTheory.ComposableArrows C n
                  ⊢ LE.le n n
                -/
abbrev right := obj' F n
                /-
                  🎉 no goals
                -/


/-- The canonical map `F.left ⟶ F.right` for `F : ComposableArrows C n`. -/
                                 /-
                                   C : Type u_1
                                   inst✝ : CategoryTheory.Category.{?u.2851, u_1} C
                                   n m : Nat
                                   F G : CategoryTheory.ComposableArrows C n
                                   ⊢ LE.le 0 n
                                 -/
                                 /-
                                   🎉 no goals
                                 -/
abbrev hom : F.left ⟶ F.right := map' F 0 n
                                 /-
                                   🎉 no goals
                                 -/


/-- The map `F.obj' i ⟶ G.obj' i` induced on `i`th objects by a morphism `F ⟶ G`
in `ComposableArrows C n` when `i` is a natural number such that `i ≤ n`. -/
@[simp]
abbrev app' (φ : F ⟶ G) (i : ℕ) (hi : i ≤ n := by valid) :
    /-
      C : Type u_1
      inst✝ : CategoryTheory.Category.{?u.3250, u_1} C
      n m : Nat
      F G : CategoryTheory.ComposableArrows C n
      φ : Quiver.Hom F G
      i : Nat
      hi : autoParam (LE.le i n) _auto✝
      ⊢ LE.le i n
    -/
    /-
      🎉 no goals
    -/
    F.obj' i ⟶ G.obj' i := φ.app _
               /-
                 🎉 no goals
               -/


@[reassoc]
lemma naturality' (φ : F ⟶ G) (i j : ℕ) (hij : i ≤ j := by valid)
    (hj : j ≤ n := by valid) :
    /-
      C : Type u_1
      inst✝ : CategoryTheory.Category.{?u.3471, u_1} C
      n m : Nat
      F G : CategoryTheory.ComposableArrows C n
      φ : Quiver.Hom F G
      i j : Nat
      hij : autoParam (LE.le i j) _auto✝
      hj : autoParam (LE.le j n) _auto✝
      ⊢ LE.le i j
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
                 /-
                   🎉 no goals
                 -/
                            /-
                              🎉 no goals
                            -/
                                       /-
                                         🎉 no goals
                                       -/
    F.map' i j ≫ app' φ j = app' φ i ≫ G.map' i j :=
                                       /-
                                         🎉 no goals
                                       -/
  φ.naturality _


/-- Constructor for `ComposableArrows C 0`. -/
@[simps!]
def mk₀ (X : C) : ComposableArrows C 0 := (Functor.const (Fin 1)).obj X


/-- The map which sends `0 : Fin 2` to `X₀` and `1` to `X₁`. -/
@[simp]
def obj : Fin 2 → C
  | ⟨0, _⟩ => X₀
  | ⟨1, _⟩  => X₁


/-- The obvious map `obj X₀ X₁ i ⟶ obj X₀ X₁ j` whenever `i j : Fin 2` satisfy `i ≤ j`. -/
@[simp]
def map : ∀ (i j : Fin 2) (_ : i ≤ j), obj X₀ X₁ i ⟶ obj X₀ X₁ j
  | ⟨0, _⟩, ⟨0, _⟩, _ => 𝟙 _
  | ⟨0, _⟩, ⟨1, _⟩, _ => f
  | ⟨1, _⟩, ⟨1, _⟩, _ => 𝟙 _


                                         /-
                                           C : Type u_1
                                           inst✝ : CategoryTheory.Category.{?u.10828, u_1} C
                                           n m : Nat
                                           F G : CategoryTheory.ComposableArrows C n
                                           X₀ X₁ : C
                                           f : Quiver.Hom X₀ X₁
                                           i : Fin 2
                                           ⊢ LE.le i i
                                         -/
lemma map_id (i : Fin 2) : map f i i (by simp) = 𝟙 _ :=
                                         /-
                                           🎉 no goals
                                         -/
  match i with
    | 0 => rfl
    | 1 => rfl


lemma map_comp {i j k : Fin 2} (hij : i ≤ j) (hjk : j ≤ k) :
    map f i k (hij.trans hjk) = map f i j hij ≫ map f j k hjk := by
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    X₀ X₁ : C
    f : Quiver.Hom X₀ X₁
    i j k : Fin 2
    hij : LE.le i j
    hjk : LE.le j k
    ⊢ Eq (CategoryTheory.ComposableArrows.Mk₁.map f i k ⋯) (CategoryTheory.Categor …
  -/
  obtain rfl | rfl : i = j ∨ j = k := by omega
    /-
      case inl
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      X₀ X₁ : C
      f : Quiver.Hom X₀ X₁
      i k : Fin 2
      hij : LE.le i i
      hjk : LE.le i k
      ⊢ Eq (CategoryTheory.ComposableArrows.Mk₁.map f i k ⋯) (CategoryTheory.Categor …
    -/
  · rw [map_id, id_comp]
    /-
      🎉 no goals
    -/
    /-
      case inr
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      X₀ X₁ : C
      f : Quiver.Hom X₀ X₁
      i j : Fin 2
      hij : LE.le i j
      hjk : LE.le j j
      ⊢ Eq (CategoryTheory.ComposableArrows.Mk₁.map f i j ⋯) (CategoryTheory.Categor …
    -/
  · rw [map_id, comp_id]
    /-
      🎉 no goals
    -/


/-- Constructor for `ComposableArrows C 1`. -/
@[simps]
def mk₁ {X₀ X₁ : C} (f : X₀ ⟶ X₁) : ComposableArrows C 1 where
  obj := Mk₁.obj X₀ X₁
  map g := Mk₁.map f _ _ (leOfHom g)
  map_id := Mk₁.map_id f
  map_comp g g' := Mk₁.map_comp f (leOfHom g) (leOfHom g')


/-- Constructor for morphisms `F ⟶ G` in `ComposableArrows C n` which takes as inputs
a family of morphisms `F.obj i ⟶ G.obj i` and the naturality condition only for the
maps in `Fin (n + 1)` given by inequalities of the form `i ≤ i + 1`. -/
@[simps]
def homMk {F G : ComposableArrows C n} (app : ∀ i, F.obj i ⟶ G.obj i)
                                 /-
                                   C : Type u_1
                                   inst✝ : CategoryTheory.Category.{?u.15169, u_1} C
                                   n m : Nat
                                   F✝ G✝ F G : CategoryTheory.ComposableArrows C n
                                   app : (i : Fin (HAdd.hAdd n 1)) → Quiver.Hom (F.obj i) (G.obj i)
                                   i : Nat
                                   hi : LT.lt i n
                                   ⊢ LE.le i (HAdd.hAdd i 1)
                                 -/
                                 /-
                                   🎉 no goals
                                 -/
                                 /-
                                   🎉 no goals
                                 -/
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
    (w : ∀ (i : ℕ) (hi : i < n), F.map' i (i + 1) ≫ app _ = app _ ≫ G.map' i (i + 1)) :
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
    F ⟶ G where
  app := app
  naturality := by
    suffices ∀ (k i j : ℕ) (hj : i + k = j) (hj' : j ≤ n),
        F.map' i j ≫ app _ = app _ ≫ G.map' i j by
      rintro ⟨i, hi⟩ ⟨j, hj⟩ hij
      have hij' := leOfHom hij
      simp only [Fin.mk_le_mk] at hij'
      obtain ⟨k, hk⟩ := Nat.le.dest hij'
      exact this k i j hk (by valid)
    /-
      C : Type u_1
      inst✝ : CategoryTheory.Category.{?u.15169, u_1} C
      n m : Nat
      F✝ G✝ F G : CategoryTheory.ComposableArrows C n
      app : (i : Fin (HAdd.hAdd n 1)) → Quiver.Hom (F.obj i) (G.obj i)
      w : ∀ (i : Nat) (hi : LT.lt i n), Eq (CategoryTheory.CategoryStruct.comp (F.ma …
      ⊢ ∀ (k i j : Nat) (hj : Eq (HAdd.hAdd i k) j) (hj' : LE.le j n), Eq (CategoryT …
    -/
    intro k
    /-
      C : Type u_1
      inst✝ : CategoryTheory.Category.{?u.15169, u_1} C
      n m : Nat
      F✝ G✝ F G : CategoryTheory.ComposableArrows C n
      app : (i : Fin (HAdd.hAdd n 1)) → Quiver.Hom (F.obj i) (G.obj i)
      w : ∀ (i : Nat) (hi : LT.lt i n), Eq (CategoryTheory.CategoryStruct.comp (F.ma …
      k : Nat
      ⊢ ∀ (i j : Nat) (hj : Eq (HAdd.hAdd i k) j) (hj' : LE.le j n), Eq (CategoryThe …
    -/
    induction' k with k hk
      /-
        case zero
        C : Type u_1
        inst✝ : CategoryTheory.Category.{?u.15169, u_1} C
        n m : Nat
        F✝ G✝ F G : CategoryTheory.ComposableArrows C n
        app : (i : Fin (HAdd.hAdd n 1)) → Quiver.Hom (F.obj i) (G.obj i)
        w : ∀ (i : Nat) (hi : LT.lt i n), Eq (CategoryTheory.CategoryStruct.comp (F.ma …
        ⊢ ∀ (i j : Nat) (hj : Eq (HAdd.hAdd i 0) j) (hj' : LE.le j n), Eq (CategoryThe …
      -/
    · intro i j hj hj'
      /-
        case zero
        C : Type u_1
        inst✝ : CategoryTheory.Category.{?u.15169, u_1} C
        n m : Nat
        F✝ G✝ F G : CategoryTheory.ComposableArrows C n
        app : (i : Fin (HAdd.hAdd n 1)) → Quiver.Hom (F.obj i) (G.obj i)
        w : ∀ (i : Nat) (hi : LT.lt i n), Eq (CategoryTheory.CategoryStruct.comp (F.ma …
        i j : Nat
        hj : Eq (HAdd.hAdd i 0) j
        hj' : LE.le j n
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map' i j ⋯ hj') (app ⟨j, ⋯⟩)) (Cat …
      -/
      simp only [add_zero] at hj
      /-
        case zero
        C : Type u_1
        inst✝ : CategoryTheory.Category.{?u.15169, u_1} C
        n m : Nat
        F✝ G✝ F G : CategoryTheory.ComposableArrows C n
        app : (i : Fin (HAdd.hAdd n 1)) → Quiver.Hom (F.obj i) (G.obj i)
        w : ∀ (i : Nat) (hi : LT.lt i n), Eq (CategoryTheory.CategoryStruct.comp (F.ma …
        i j : Nat
        hj✝ : Eq (HAdd.hAdd i 0) j
        hj' : LE.le j n
        hj : Eq i j
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map' i j ⋯ hj') (app ⟨j, ⋯⟩)) (Cat …
      -/
      obtain rfl := hj
      /-
        case zero
        C : Type u_1
        inst✝ : CategoryTheory.Category.{?u.15169, u_1} C
        n m : Nat
        F✝ G✝ F G : CategoryTheory.ComposableArrows C n
        app : (i : Fin (HAdd.hAdd n 1)) → Quiver.Hom (F.obj i) (G.obj i)
        w : ∀ (i : Nat) (hi : LT.lt i n), Eq (CategoryTheory.CategoryStruct.comp (F.ma …
        i : Nat
        hj : Eq (HAdd.hAdd i 0) i
        hj' : LE.le i n
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map' i i ⋯ hj') (app ⟨i, ⋯⟩)) (Cat …
      -/
      rw [F.map'_self i, G.map'_self i, id_comp, comp_id]
      /-
        🎉 no goals
      -/
      /-
        case succ
        C : Type u_1
        inst✝ : CategoryTheory.Category.{?u.15169, u_1} C
        n m : Nat
        F✝ G✝ F G : CategoryTheory.ComposableArrows C n
        app : (i : Fin (HAdd.hAdd n 1)) → Quiver.Hom (F.obj i) (G.obj i)
        w : ∀ (i : Nat) (hi : LT.lt i n), Eq (CategoryTheory.CategoryStruct.comp (F.ma …
        k : Nat
        hk : ∀ (i j : Nat) (hj : Eq (HAdd.hAdd i k) j) (hj' : LE.le j n), Eq (Category …
        ⊢ ∀ (i j : Nat) (hj : Eq (HAdd.hAdd i (HAdd.hAdd k 1)) j) (hj' : LE.le j n), E …
      -/
    · intro i j hj hj'
      /-
        case succ
        C : Type u_1
        inst✝ : CategoryTheory.Category.{?u.15169, u_1} C
        n m : Nat
        F✝ G✝ F G : CategoryTheory.ComposableArrows C n
        app : (i : Fin (HAdd.hAdd n 1)) → Quiver.Hom (F.obj i) (G.obj i)
        w : ∀ (i : Nat) (hi : LT.lt i n), Eq (CategoryTheory.CategoryStruct.comp (F.ma …
        k : Nat
        hk : ∀ (i j : Nat) (hj : Eq (HAdd.hAdd i k) j) (hj' : LE.le j n), Eq (Category …
        i j : Nat
        hj : Eq (HAdd.hAdd i (HAdd.hAdd k 1)) j
        hj' : LE.le j n
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map' i j ⋯ hj') (app ⟨j, ⋯⟩)) (Cat …
      -/
      rw [← add_assoc] at hj
      /-
        case succ
        C : Type u_1
        inst✝ : CategoryTheory.Category.{?u.15169, u_1} C
        n m : Nat
        F✝ G✝ F G : CategoryTheory.ComposableArrows C n
        app : (i : Fin (HAdd.hAdd n 1)) → Quiver.Hom (F.obj i) (G.obj i)
        w : ∀ (i : Nat) (hi : LT.lt i n), Eq (CategoryTheory.CategoryStruct.comp (F.ma …
        k : Nat
        hk : ∀ (i j : Nat) (hj : Eq (HAdd.hAdd i k) j) (hj' : LE.le j n), Eq (Category …
        i j : Nat
        hj✝ : Eq (HAdd.hAdd i (HAdd.hAdd k 1)) j
        hj : Eq (HAdd.hAdd (HAdd.hAdd i k) 1) j
        hj' : LE.le j n
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map' i j ⋯ hj') (app ⟨j, ⋯⟩)) (Cat …
      -/
      subst hj
      rw [F.map'_comp i (i + k) (i + k + 1), G.map'_comp i (i + k) (i + k + 1), assoc,
        w (i + k) (by valid), reassoc_of% (hk i (i + k) rfl (by valid))]


/-- Constructor for isomorphisms `F ≅ G` in `ComposableArrows C n` which takes as inputs
a family of isomorphisms `F.obj i ≅ G.obj i` and the naturality condition only for the
maps in `Fin (n + 1)` given by inequalities of the form `i ≤ i + 1`. -/
@[simps]
def isoMk {F G : ComposableArrows C n} (app : ∀ i, F.obj i ≅ G.obj i)
    (w : ∀ (i : ℕ) (hi : i < n),
      /-
        C : Type u_1
        inst✝ : CategoryTheory.Category.{?u.42165, u_1} C
        n m : Nat
        F✝ G✝ F G : CategoryTheory.ComposableArrows C n
        app : (i : Fin (HAdd.hAdd n 1)) → CategoryTheory.Iso (F.obj i) (G.obj i)
        i : Nat
        hi : LT.lt i n
        ⊢ LE.le i (HAdd.hAdd i 1)
      -/
      /-
        🎉 no goals
      -/
      /-
        🎉 no goals
      -/
                                                     /-
                                                       🎉 no goals
                                                     -/
      F.map' i (i + 1) ≫ (app _).hom = (app _).hom ≫ G.map' i (i + 1)) :
                                                     /-
                                                       🎉 no goals
                                                     -/
    F ≅ G where
  hom := homMk (fun i => (app i).hom) w
  inv := homMk (fun i => (app i).inv) (fun i hi => by
    /-
      C : Type u_1
      inst✝ : CategoryTheory.Category.{?u.42165, u_1} C
      n m : Nat
      F✝ G✝ F G : CategoryTheory.ComposableArrows C n
      app : (i : Fin (HAdd.hAdd n 1)) → CategoryTheory.Iso (F.obj i) (G.obj i)
      w : ∀ (i : Nat) (hi : LT.lt i n), Eq (CategoryTheory.CategoryStruct.comp (F.ma …
      i : Nat
      hi : LT.lt i n
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map' i (HAdd.hAdd i 1) ⋯ hi) ((fun …
    -/
    dsimp only
    rw [← cancel_epi ((app _).hom), ← reassoc_of% (w i hi), Iso.hom_inv_id, comp_id,
      Iso.hom_inv_id_assoc])


lemma ext {F G : ComposableArrows C n} (h : ∀ i, F.obj i = G.obj i)
                                 /-
                                   C : Type u_1
                                   inst✝ : CategoryTheory.Category.{?u.50816, u_1} C
                                   n m : Nat
                                   F✝ G✝ F G : CategoryTheory.ComposableArrows C n
                                   h : ∀ (i : Fin (HAdd.hAdd n 1)), Eq (F.obj i) (G.obj i)
                                   i : Nat
                                   hi : LT.lt i n
                                   ⊢ LE.le i (HAdd.hAdd i 1)
                                 -/
                                 /-
                                   🎉 no goals
                                 -/
    (w : ∀ (i : ℕ) (hi : i < n), F.map' i (i + 1) =
                                 /-
                                   🎉 no goals
                                 -/
                      /-
                        C : Type u_1
                        inst✝ : CategoryTheory.Category.{?u.50816, u_1} C
                        n m : Nat
                        F✝ G✝ F G : CategoryTheory.ComposableArrows C n
                        h : ∀ (i : Fin (HAdd.hAdd n 1)), Eq (F.obj i) (G.obj i)
                        i : Nat
                        hi : LT.lt i n
                        ⊢ LE.le i (HAdd.hAdd i 1)
                      -/
                      /-
                        🎉 no goals
                      -/
      eqToHom (h _) ≫ G.map' i (i + 1) ≫ eqToHom (h _).symm) : F = G :=
                      /-
                        🎉 no goals
                      -/
  Functor.ext_of_iso
                                                    /-
                                                      C : Type u_1
                                                      inst✝ : CategoryTheory.Category.{u_2, u_1} C
                                                      n : Nat
                                                      F G : CategoryTheory.ComposableArrows C n
                                                      h : ∀ (i : Fin (HAdd.hAdd n 1)), Eq (F.obj i) (G.obj i)
                                                      w : ∀ (i : Nat) (hi : LT.lt i n), Eq (F.map' i (HAdd.hAdd i 1) ⋯ hi) (Category …
                                                      i : Nat
                                                      hi : LT.lt i n
                                                      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map' i (HAdd.hAdd i 1) ⋯ hi) ((fun …
                                                    -/
    (isoMk (fun i => eqToIso (h i)) (fun i hi => by simp [w i hi])) h (fun _ => rfl)
                                                    /-
                                                      🎉 no goals
                                                    -/


/-- Constructor for morphisms in `ComposableArrows C 0`. -/
@[simps!]
                                             /-
                                               C : Type u_1
                                               inst✝ : CategoryTheory.Category.{?u.61143, u_1} C
                                               n m : Nat
                                               F✝ G✝ : CategoryTheory.ComposableArrows C n
                                               F G : CategoryTheory.ComposableArrows C 0
                                               ⊢ LE.le 0 0
                                             -/
                                             /-
                                               🎉 no goals
                                             -/
def homMk₀ {F G : ComposableArrows C 0} (f : F.obj' 0 ⟶ G.obj' 0) : F ⟶ G :=
                                                        /-
                                                          🎉 no goals
                                                        -/
  homMk (fun i => match i with
                                   /-
                                     C : Type u_1
                                     inst✝ : CategoryTheory.Category.{?u.61143, u_1} C
                                     n m : Nat
                                     F✝ G✝ : CategoryTheory.ComposableArrows C n
                                     F G : CategoryTheory.ComposableArrows C 0
                                     f : Quiver.Hom (F.obj' 0 ⋯) (G.obj' 0 ⋯)
                                     i : Nat
                                     hi : LT.lt i 0
                                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map' i (HAdd.hAdd i 1) ⋯ hi) ((fun …
                                   -/
    | ⟨0, _⟩ => f) (fun i hi => by simp at hi)
                                   /-
                                     🎉 no goals
                                   -/


@[ext]
lemma hom_ext₀ {F G : ComposableArrows C 0} {φ φ' : F ⟶ G}
         /-
           C : Type u_1
           inst✝ : CategoryTheory.Category.{?u.62072, u_1} C
           n m : Nat
           F✝ G✝ : CategoryTheory.ComposableArrows C n
           F G : CategoryTheory.ComposableArrows C 0
           φ φ' : Quiver.Hom F G
           ⊢ LE.le 0 0
         -/
         /-
           🎉 no goals
         -/
    (h : app' φ 0 = app' φ' 0) :
                    /-
                      🎉 no goals
                    -/
    φ = φ' := by
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    F G : CategoryTheory.ComposableArrows C 0
    φ φ' : Quiver.Hom F G
    h : Eq (CategoryTheory.ComposableArrows.app' φ 0 ⋯) (CategoryTheory.Composable …
    ⊢ Eq φ φ'
  -/
  ext i
  /-
    case w.h
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    F G : CategoryTheory.ComposableArrows C 0
    φ φ' : Quiver.Hom F G
    h : Eq (CategoryTheory.ComposableArrows.app' φ 0 ⋯) (CategoryTheory.Composable …
    i : Fin (HAdd.hAdd 0 1)
    ⊢ Eq (φ.app i) (φ'.app i)
  -/
  fin_cases i
  /-
    case w.h.«0»
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    F G : CategoryTheory.ComposableArrows C 0
    φ φ' : Quiver.Hom F G
    h : Eq (CategoryTheory.ComposableArrows.app' φ 0 ⋯) (CategoryTheory.Composable …
    ⊢ Eq (φ.app ((fun i => i) ⟨0, ⋯⟩)) (φ'.app ((fun i => i) ⟨0, ⋯⟩))
  -/
  exact h
  /-
    🎉 no goals
  -/


/-- Constructor for isomorphisms in `ComposableArrows C 0`. -/
@[simps!]
                                             /-
                                               C : Type u_1
                                               inst✝ : CategoryTheory.Category.{?u.62938, u_1} C
                                               n m : Nat
                                               F✝ G✝ : CategoryTheory.ComposableArrows C n
                                               F G : CategoryTheory.ComposableArrows C 0
                                               ⊢ LE.le 0 0
                                             -/
                                             /-
                                               🎉 no goals
                                             -/
def isoMk₀ {F G : ComposableArrows C 0} (e : F.obj' 0 ≅ G.obj' 0) : F ≅ G where
                                                        /-
                                                          🎉 no goals
                                                        -/
  hom := homMk₀ e.hom
  inv := homMk₀ e.inv


                                             /-
                                               C : Type u_1
                                               inst✝ : CategoryTheory.Category.{?u.64937, u_1} C
                                               n m : Nat
                                               F✝ G✝ : CategoryTheory.ComposableArrows C n
                                               F G : CategoryTheory.ComposableArrows C 0
                                               ⊢ LE.le 0 0
                                             -/
lemma ext₀ {F G : ComposableArrows C 0} (h : F.obj' 0 = G.obj 0) : F = G :=
                                             /-
                                               🎉 no goals
                                             -/
  ext (fun i => match i with
                                   /-
                                     C : Type u_1
                                     inst✝ : CategoryTheory.Category.{u_2, u_1} C
                                     F G : CategoryTheory.ComposableArrows C 0
                                     h : Eq (F.obj' 0 ⋯) (G.obj 0)
                                     i : Nat
                                     hi : LT.lt i 0
                                     ⊢ Eq (F.map' i (HAdd.hAdd i 1) ⋯ hi) (CategoryTheory.CategoryStruct.comp (Cate …
                                   -/
    | ⟨0, _⟩ => h) (fun i hi => by simp at hi)
                                   /-
                                     🎉 no goals
                                   -/


lemma mk₀_surjective (F : ComposableArrows C 0) : ∃ (X : C), F = mk₀ X :=
   /-
     C : Type u_1
     inst✝ : CategoryTheory.Category.{u_2, u_1} C
     F : CategoryTheory.ComposableArrows C 0
     ⊢ LE.le 0 0
   -/
  ⟨F.obj' 0, ext₀ rfl⟩
   /-
     🎉 no goals
   -/


/-- Constructor for morphisms in `ComposableArrows C 1`. -/
@[simps!]
def homMk₁ {F G : ComposableArrows C 1}
            /-
              C : Type u_1
              inst✝ : CategoryTheory.Category.{?u.65994, u_1} C
              n m : Nat
              F✝ G✝ : CategoryTheory.ComposableArrows C n
              F G : CategoryTheory.ComposableArrows C 1
              ⊢ LE.le 0 1
            -/
            /-
              🎉 no goals
            -/
                       /-
                         🎉 no goals
                       -/
                                          /-
                                            🎉 no goals
                                          -/
    (left : F.obj' 0 ⟶ G.obj' 0) (right : F.obj' 1 ⟶ G.obj' 1)
                                                     /-
                                                       🎉 no goals
                                                     -/
         /-
           C : Type u_1
           inst✝ : CategoryTheory.Category.{?u.65994, u_1} C
           n m : Nat
           F✝ G✝ : CategoryTheory.ComposableArrows C n
           F G : CategoryTheory.ComposableArrows C 1
           left : Quiver.Hom (F.obj' 0 ⋯) (G.obj' 0 ⋯)
           right : Quiver.Hom (F.obj' 1 ⋯) (G.obj' 1 ⋯)
           ⊢ LE.le 0 1
         -/
         /-
           🎉 no goals
         -/
         /-
           🎉 no goals
         -/
                                     /-
                                       🎉 no goals
                                     -/
    (w : F.map' 0 1 ≫ right = left ≫ G.map' 0 1 := by aesop_cat) :
                                     /-
                                       🎉 no goals
                                     -/
    F ⟶ G :=
  homMk (fun i => match i with
      | ⟨0, _⟩ => left
      | ⟨1, _⟩ => right) (by
          /-
            C : Type u_1
            inst✝ : CategoryTheory.Category.{?u.65994, u_1} C
            n m : Nat
            F✝ G✝ : CategoryTheory.ComposableArrows C n
            F G : CategoryTheory.ComposableArrows C 1
            left : Quiver.Hom (F.obj' 0 ⋯) (G.obj' 0 ⋯)
            right : Quiver.Hom (F.obj' 1 ⋯) (G.obj' 1 ⋯)
            w : autoParam (Eq (CategoryTheory.CategoryStruct.comp (F.map' 0 1 ⋯ ⋯) right)  …
            ⊢ ∀ (i : Nat) (hi : LT.lt i 1), Eq (CategoryTheory.CategoryStruct.comp (F.map' …
          -/
          intro i hi
          /-
            C : Type u_1
            inst✝ : CategoryTheory.Category.{?u.65994, u_1} C
            n m : Nat
            F✝ G✝ : CategoryTheory.ComposableArrows C n
            F G : CategoryTheory.ComposableArrows C 1
            left : Quiver.Hom (F.obj' 0 ⋯) (G.obj' 0 ⋯)
            right : Quiver.Hom (F.obj' 1 ⋯) (G.obj' 1 ⋯)
            w : autoParam (Eq (CategoryTheory.CategoryStruct.comp (F.map' 0 1 ⋯ ⋯) right)  …
            i : Nat
            hi : LT.lt i 1
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map' i (HAdd.hAdd i 1) ⋯ hi) ((fun …
          -/
          obtain rfl : i = 0 := by simpa using hi
          /-
            C : Type u_1
            inst✝ : CategoryTheory.Category.{?u.65994, u_1} C
            n m : Nat
            F✝ G✝ : CategoryTheory.ComposableArrows C n
            F G : CategoryTheory.ComposableArrows C 1
            left : Quiver.Hom (F.obj' 0 ⋯) (G.obj' 0 ⋯)
            right : Quiver.Hom (F.obj' 1 ⋯) (G.obj' 1 ⋯)
            w : autoParam (Eq (CategoryTheory.CategoryStruct.comp (F.map' 0 1 ⋯ ⋯) right)  …
            hi : LT.lt 0 1
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map' 0 (HAdd.hAdd 0 1) ⋯ hi) ((fun …
          -/
          exact w)
          /-
            🎉 no goals
          -/


@[ext]
lemma hom_ext₁ {F G : ComposableArrows C 1} {φ φ' : F ⟶ G}
          /-
            C : Type u_1
            inst✝ : CategoryTheory.Category.{?u.69782, u_1} C
            n m : Nat
            F✝ G✝ : CategoryTheory.ComposableArrows C n
            F G : CategoryTheory.ComposableArrows C 1
            φ φ' : Quiver.Hom F G
            ⊢ LE.le 0 1
          -/
          /-
            🎉 no goals
          -/
                     /-
                       🎉 no goals
                     -/
                                      /-
                                        🎉 no goals
                                      -/
    (h₀ : app' φ 0 = app' φ' 0) (h₁ : app' φ 1 = app' φ' 1) :
                                                 /-
                                                   🎉 no goals
                                                 -/
    φ = φ' := by
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    F G : CategoryTheory.ComposableArrows C 1
    φ φ' : Quiver.Hom F G
    h₀ : Eq (CategoryTheory.ComposableArrows.app' φ 0 ⋯) (CategoryTheory.Composabl …
    h₁ : Eq (CategoryTheory.ComposableArrows.app' φ 1 ⋯) (CategoryTheory.Composabl …
    ⊢ Eq φ φ'
  -/
  ext i
  match i with
    | 0 => exact h₀
    | 1 => exact h₁


/-- Constructor for isomorphisms in `ComposableArrows C 1`. -/
@[simps!]
def isoMk₁ {F G : ComposableArrows C 1}
            /-
              C : Type u_1
              inst✝ : CategoryTheory.Category.{?u.71517, u_1} C
              n m : Nat
              F✝ G✝ : CategoryTheory.ComposableArrows C n
              F G : CategoryTheory.ComposableArrows C 1
              ⊢ LE.le 0 1
            -/
            /-
              🎉 no goals
            -/
                       /-
                         🎉 no goals
                       -/
                                          /-
                                            🎉 no goals
                                          -/
    (left : F.obj' 0 ≅ G.obj' 0) (right : F.obj' 1 ≅ G.obj' 1)
                                                     /-
                                                       🎉 no goals
                                                     -/
         /-
           C : Type u_1
           inst✝ : CategoryTheory.Category.{?u.71517, u_1} C
           n m : Nat
           F✝ G✝ : CategoryTheory.ComposableArrows C n
           F G : CategoryTheory.ComposableArrows C 1
           left : CategoryTheory.Iso (F.obj' 0 ⋯) (G.obj' 0 ⋯)
           right : CategoryTheory.Iso (F.obj' 1 ⋯) (G.obj' 1 ⋯)
           ⊢ LE.le 0 1
         -/
         /-
           🎉 no goals
         -/
         /-
           🎉 no goals
         -/
                                             /-
                                               🎉 no goals
                                             -/
    (w : F.map' 0 1 ≫ right.hom = left.hom ≫ G.map' 0 1 := by aesop_cat) :
                                             /-
                                               🎉 no goals
                                             -/
    F ≅ G where
  hom := homMk₁ left.hom right.hom w
  inv := homMk₁ left.inv right.inv (by
    /-
      C : Type u_1
      inst✝ : CategoryTheory.Category.{?u.71517, u_1} C
      n m : Nat
      F✝ G✝ : CategoryTheory.ComposableArrows C n
      F G : CategoryTheory.ComposableArrows C 1
      left : CategoryTheory.Iso (F.obj' 0 ⋯) (G.obj' 0 ⋯)
      right : CategoryTheory.Iso (F.obj' 1 ⋯) (G.obj' 1 ⋯)
      w : autoParam (Eq (CategoryTheory.CategoryStruct.comp (F.map' 0 1 ⋯ ⋯) right.h …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map' 0 1 ⋯ ⋯) right.inv) (Category …
    -/
    rw [← cancel_mono right.hom, assoc, assoc, w, right.inv_hom_id, left.inv_hom_id_assoc]
    /-
      C : Type u_1
      inst✝ : CategoryTheory.Category.{?u.71517, u_1} C
      n m : Nat
      F✝ G✝ : CategoryTheory.ComposableArrows C n
      F G : CategoryTheory.ComposableArrows C 1
      left : CategoryTheory.Iso (F.obj' 0 ⋯) (G.obj' 0 ⋯)
      right : CategoryTheory.Iso (F.obj' 1 ⋯) (G.obj' 1 ⋯)
      w : autoParam (Eq (CategoryTheory.CategoryStruct.comp (F.map' 0 1 ⋯ ⋯) right.h …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map' 0 1 ⋯ ⋯) (CategoryTheory.Cate …
    -/
    apply comp_id)
    /-
      🎉 no goals
    -/


                                                /-
                                                  C : Type u_1
                                                  inst✝ : CategoryTheory.Category.{?u.81278, u_1} C
                                                  n m : Nat
                                                  F✝ G : CategoryTheory.ComposableArrows C n
                                                  F : CategoryTheory.ComposableArrows C 1
                                                  ⊢ LE.le 0 1
                                                -/
                                                /-
                                                  🎉 no goals
                                                -/
lemma map'_eq_hom₁ (F : ComposableArrows C 1) : F.map' 0 1 = F.hom := rfl
                                                /-
                                                  🎉 no goals
                                                -/


lemma ext₁ {F G : ComposableArrows C 1}
    (left : F.left = G.left) (right : F.right = G.right)
    (w : F.hom = eqToHom left ≫ G.hom ≫ eqToHom right.symm) : F = G :=
                                                                /-
                                                                  C : Type u_1
                                                                  inst✝ : CategoryTheory.Category.{u_2, u_1} C
                                                                  F G : CategoryTheory.ComposableArrows C 1
                                                                  left : Eq F.left G.left
                                                                  right : Eq F.right G.right
                                                                  w : Eq F.hom (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom left) …
                                                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map' 0 1 ⋯ ⋯) (CategoryTheory.eqTo …
                                                                -/
  Functor.ext_of_iso (isoMk₁ (eqToIso left) (eqToIso right) (by simp [map'_eq_hom₁, w]))
                                                                /-
                                                                  🎉 no goals
                                                                -/
                 /-
                   C : Type u_1
                   inst✝ : CategoryTheory.Category.{u_2, u_1} C
                   F G : CategoryTheory.ComposableArrows C 1
                   left : Eq F.left G.left
                   right : Eq F.right G.right
                   w : Eq F.hom (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom left) …
                   i : Fin (HAdd.hAdd 1 1)
                   ⊢ Eq (F.obj i) (G.obj i)
                 -/
                                 /-
                                   🎉 no goals
                                 -/
    (fun i => by fin_cases i <;> assumption)
                                 /-
                                   🎉 no goals
                                 -/
                 /-
                   C : Type u_1
                   inst✝ : CategoryTheory.Category.{u_2, u_1} C
                   F G : CategoryTheory.ComposableArrows C 1
                   left : Eq F.left G.left
                   right : Eq F.right G.right
                   w : Eq F.hom (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom left) …
                   i : Fin (HAdd.hAdd 1 1)
                   ⊢ Eq ((CategoryTheory.ComposableArrows.isoMk₁ (CategoryTheory.eqToIso left) (C …
                 -/
                                 /-
                                   🎉 no goals
                                 -/
    (fun i => by fin_cases i <;> rfl)
                                 /-
                                   🎉 no goals
                                 -/


lemma mk₁_surjective (X : ComposableArrows C 1) : ∃ (X₀ X₁ : C) (f : X₀ ⟶ X₁), X = mk₁ f :=
         /-
           C : Type u_1
           inst✝ : CategoryTheory.Category.{u_2, u_1} C
           X : CategoryTheory.ComposableArrows C 1
           ⊢ LE.le 0 1
         -/
         /-
           🎉 no goals
         -/
         /-
           🎉 no goals
         -/
  ⟨_, _, X.map' 0 1, ext₁ rfl rfl (by simp)⟩
                                      /-
                                        🎉 no goals
                                      -/


/-- The map `Fin (n + 1 + 1) → C` which "shifts" `F.obj'` to the right and inserts `X` in
the zeroth position. -/
def obj : Fin (n + 1 + 1) → C
  | ⟨0, _⟩ => X
                   /-
                     C : Type u_1
                     inst✝ : CategoryTheory.Category.{?u.86592, u_1} C
                     n m : Nat
                     F G : CategoryTheory.ComposableArrows C n
                     X : C
                     i : Nat
                     hi : LT.lt (HAdd.hAdd i 1) (HAdd.hAdd (HAdd.hAdd n 1) 1)
                     ⊢ LE.le i n
                   -/
  | ⟨i + 1, hi⟩ => F.obj' i
                   /-
                     🎉 no goals
                   -/


@[simp]
lemma obj_zero : obj F X 0 = X := rfl


@[simp]
                            /-
                              C : Type u_1
                              inst✝ : CategoryTheory.Category.{?u.87343, u_1} C
                              n m : Nat
                              F G : CategoryTheory.ComposableArrows C n
                              X : C
                              ⊢ LE.le 0 n
                            -/
lemma obj_one : obj F X 1 = F.obj' 0 := rfl
                            /-
                              🎉 no goals
                            -/


@[simp]
                                                                        /-
                                                                          C : Type u_1
                                                                          inst✝ : CategoryTheory.Category.{?u.87761, u_1} C
                                                                          n m : Nat
                                                                          F G : CategoryTheory.ComposableArrows C n
                                                                          X : C
                                                                          i : Nat
                                                                          hi : LT.lt (HAdd.hAdd i 1) (HAdd.hAdd (HAdd.hAdd n 1) 1)
                                                                          ⊢ LE.le i n
                                                                        -/
lemma obj_succ (i : ℕ) (hi : i + 1 < n + 1 + 1) : obj F X ⟨i + 1, hi⟩ = F.obj' i := rfl
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


/-- Auxiliary definition for the action on maps of the functor `F.precomp f`.
It sends `0 ≤ 1` to `f` and `i + 1 ≤ j + 1` to `F.map' i j`. -/
def map : ∀ (i j : Fin (n + 1 + 1)) (_ : i ≤ j), obj F X i ⟶ obj F X j
  | ⟨0, _⟩, ⟨0, _⟩, _ => 𝟙 X
  | ⟨0, _⟩, ⟨1, _⟩, _ => f
                                  /-
                                    C : Type u_1
                                    inst✝ : CategoryTheory.Category.{?u.88427, u_1} C
                                    n m : Nat
                                    F G : CategoryTheory.ComposableArrows C n
                                    X : C
                                    f : Quiver.Hom X F.left
                                    isLt✝ : LT.lt 0 (HAdd.hAdd (HAdd.hAdd n 1) 1)
                                    j : Nat
                                    hj : LT.lt (HAdd.hAdd j 2) (HAdd.hAdd (HAdd.hAdd n 1) 1)
                                    x✝ : LE.le ⟨0, isLt✝⟩ ⟨HAdd.hAdd j 2, hj⟩
                                    ⊢ LE.le 0 (HAdd.hAdd j 1)
                                  -/
                                  /-
                                    🎉 no goals
                                  -/
  | ⟨0, _⟩, ⟨j + 2, hj⟩, _ => f ≫ F.map' 0 (j + 1)
                                  /-
                                    🎉 no goals
                                  -/
                                                    /-
                                                      C : Type u_1
                                                      inst✝ : CategoryTheory.Category.{?u.88427, u_1} C
                                                      n m : Nat
                                                      F G : CategoryTheory.ComposableArrows C n
                                                      X : C
                                                      f : Quiver.Hom X F.left
                                                      i : Nat
                                                      hi : LT.lt (HAdd.hAdd i 1) (HAdd.hAdd (HAdd.hAdd n 1) 1)
                                                      j : Nat
                                                      hj : LT.lt (HAdd.hAdd j 1) (HAdd.hAdd (HAdd.hAdd n 1) 1)
                                                      hij : LE.le ⟨HAdd.hAdd i 1, hi⟩ ⟨HAdd.hAdd j 1, hj⟩
                                                      ⊢ LE.le i j
                                                    -/
                                                    /-
                                                      🎉 no goals
                                                    -/
  | ⟨i + 1, hi⟩, ⟨j + 1, hj⟩, hij => F.map' i j (by simpa using hij)
                                     /-
                                       🎉 no goals
                                     -/


@[simp]
                                      /-
                                        C : Type u_1
                                        inst✝ : CategoryTheory.Category.{?u.91793, u_1} C
                                        n m : Nat
                                        F G : CategoryTheory.ComposableArrows C n
                                        X : C
                                        f : Quiver.Hom X F.left
                                        ⊢ LE.le 0 0
                                      -/
lemma map_zero_zero : map F f 0 0 (by simp) = 𝟙 X := rfl
                                      /-
                                        🎉 no goals
                                      -/


@[simp]
                                    /-
                                      C : Type u_1
                                      inst✝ : CategoryTheory.Category.{?u.92120, u_1} C
                                      n m : Nat
                                      F G : CategoryTheory.ComposableArrows C n
                                      X : C
                                      f : Quiver.Hom X F.left
                                      ⊢ LE.le 1 1
                                    -/
lemma map_one_one : map F f 1 1 (by simp) = F.map (𝟙 _) := rfl
                                    /-
                                      🎉 no goals
                                    -/


@[simp]
                                     /-
                                       C : Type u_1
                                       inst✝ : CategoryTheory.Category.{?u.93108, u_1} C
                                       n m : Nat
                                       F G : CategoryTheory.ComposableArrows C n
                                       X : C
                                       f : Quiver.Hom X F.left
                                       ⊢ LE.le 0 1
                                     -/
lemma map_zero_one : map F f 0 1 (by simp) = f := rfl
                                     /-
                                       🎉 no goals
                                     -/


@[simp]
                                           /-
                                             C : Type u_1
                                             inst✝ : CategoryTheory.Category.{?u.93630, u_1} C
                                             n m : Nat
                                             F G : CategoryTheory.ComposableArrows C n
                                             X : C
                                             f : Quiver.Hom X F.left
                                             ⊢ LT.lt (HAdd.hAdd 0 1) (HAdd.hAdd (HAdd.hAdd n 1) 1)
                                           -/
                                           /-
                                             🎉 no goals
                                           -/
lemma map_zero_one' : map F f 0 ⟨0 + 1, by simp⟩ (by simp) = f := rfl
                                                     /-
                                                       🎉 no goals
                                                     -/


@[simp]
lemma map_zero_succ_succ (j : ℕ) (hj : j + 2 < n + 1 + 1) :
                              /-
                                C : Type u_1
                                inst✝ : CategoryTheory.Category.{?u.95033, u_1} C
                                n m : Nat
                                F G : CategoryTheory.ComposableArrows C n
                                X : C
                                f : Quiver.Hom X F.left
                                j : Nat
                                hj : LT.lt (HAdd.hAdd j 2) (HAdd.hAdd (HAdd.hAdd n 1) 1)
                                ⊢ LE.le 0 ⟨HAdd.hAdd j 2, hj⟩
                              -/
                              /-
                                🎉 no goals
                              -/
                                          /-
                                            🎉 no goals
                                          -/
    map F f 0 ⟨j + 2, hj⟩ (by simp) = f ≫ F.map' 0 (j+1) := rfl
                                          /-
                                            🎉 no goals
                                          -/


@[simp]
lemma map_succ_succ (i j : ℕ) (hi : i + 1 < n + 1 + 1) (hj : j + 1 < n + 1 + 1)
    (hij : i + 1 ≤ j + 1) :
                                          /-
                                            C : Type u_1
                                            inst✝ : CategoryTheory.Category.{?u.96205, u_1} C
                                            n m : Nat
                                            F G : CategoryTheory.ComposableArrows C n
                                            X : C
                                            f : Quiver.Hom X F.left
                                            i j : Nat
                                            hi : LT.lt (HAdd.hAdd i 1) (HAdd.hAdd (HAdd.hAdd n 1) 1)
                                            hj : LT.lt (HAdd.hAdd j 1) (HAdd.hAdd (HAdd.hAdd n 1) 1)
                                            hij : LE.le (HAdd.hAdd i 1) (HAdd.hAdd j 1)
                                            ⊢ LE.le i j
                                          -/
                                          /-
                                            🎉 no goals
                                          -/
    map F f ⟨i + 1, hi⟩ ⟨j + 1, hj⟩ hij = F.map' i j := rfl
                                          /-
                                            🎉 no goals
                                          -/


@[simp]
lemma map_one_succ (j : ℕ) (hj : j + 1 < n + 1 + 1) :
                              /-
                                C : Type u_1
                                inst✝ : CategoryTheory.Category.{?u.97392, u_1} C
                                n m : Nat
                                F G : CategoryTheory.ComposableArrows C n
                                X : C
                                f : Quiver.Hom X F.left
                                j : Nat
                                hj : LT.lt (HAdd.hAdd j 1) (HAdd.hAdd (HAdd.hAdd n 1) 1)
                                ⊢ LE.le 1 ⟨HAdd.hAdd j 1, hj⟩
                              -/
                              /-
                                🎉 no goals
                              -/
                                                   /-
                                                     🎉 no goals
                                                   -/
    map F f 1 ⟨j + 1, hj⟩ (by simp [Fin.le_def]) = F.map' 0 j := rfl
                                                   /-
                                                     🎉 no goals
                                                   -/


                                                     /-
                                                       C : Type u_1
                                                       inst✝ : CategoryTheory.Category.{?u.98885, u_1} C
                                                       n m : Nat
                                                       F G : CategoryTheory.ComposableArrows C n
                                                       X : C
                                                       f : Quiver.Hom X F.left
                                                       i : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1)
                                                       ⊢ LE.le i i
                                                     -/
lemma map_id (i : Fin (n + 1 + 1)) : map F f i i (by simp) = 𝟙 _ := by
                                                     /-
                                                       🎉 no goals
                                                     -/
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    n : Nat
    F : CategoryTheory.ComposableArrows C n
    X : C
    f : Quiver.Hom X F.left
    i : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1)
    ⊢ Eq (CategoryTheory.ComposableArrows.Precomp.map F f i i ⋯) (CategoryTheory.C …
  -/
  obtain ⟨i, hi⟩ := i
  /-
    case mk
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    n : Nat
    F : CategoryTheory.ComposableArrows C n
    X : C
    f : Quiver.Hom X F.left
    i : Nat
    hi : LT.lt i (HAdd.hAdd (HAdd.hAdd n 1) 1)
    ⊢ Eq (CategoryTheory.ComposableArrows.Precomp.map F f ⟨i, hi⟩ ⟨i, hi⟩ ⋯) (Cate …
  -/
  cases i
    /-
      case mk.zero
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      n : Nat
      F : CategoryTheory.ComposableArrows C n
      X : C
      f : Quiver.Hom X F.left
      hi : LT.lt 0 (HAdd.hAdd (HAdd.hAdd n 1) 1)
      ⊢ Eq (CategoryTheory.ComposableArrows.Precomp.map F f ⟨0, hi⟩ ⟨0, hi⟩ ⋯) (Cate …
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case mk.succ
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      n : Nat
      F : CategoryTheory.ComposableArrows C n
      X : C
      f : Quiver.Hom X F.left
      n✝ : Nat
      hi : LT.lt (HAdd.hAdd n✝ 1) (HAdd.hAdd (HAdd.hAdd n 1) 1)
      ⊢ Eq (CategoryTheory.ComposableArrows.Precomp.map F f ⟨HAdd.hAdd n✝ 1, hi⟩ ⟨HA …
    -/
  · apply F.map_id
    /-
      🎉 no goals
    -/


lemma map_comp {i j k : Fin (n + 1 + 1)} (hij : i ≤ j) (hjk : j ≤ k) :
    map F f i k (hij.trans hjk) = map F f i j hij ≫ map F f j k hjk := by
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    n : Nat
    F : CategoryTheory.ComposableArrows C n
    X : C
    f : Quiver.Hom X F.left
    i j k : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1)
    hij : LE.le i j
    hjk : LE.le j k
    ⊢ Eq (CategoryTheory.ComposableArrows.Precomp.map F f i k ⋯) (CategoryTheory.C …
  -/
  obtain ⟨i, hi⟩ := i
  /-
    case mk
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    n : Nat
    F : CategoryTheory.ComposableArrows C n
    X : C
    f : Quiver.Hom X F.left
    j k : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1)
    hjk : LE.le j k
    i : Nat
    hi : LT.lt i (HAdd.hAdd (HAdd.hAdd n 1) 1)
    hij : LE.le ⟨i, hi⟩ j
    ⊢ Eq (CategoryTheory.ComposableArrows.Precomp.map F f ⟨i, hi⟩ k ⋯) (CategoryTh …
  -/
  obtain ⟨j, hj⟩ := j
  /-
    case mk.mk
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    n : Nat
    F : CategoryTheory.ComposableArrows C n
    X : C
    f : Quiver.Hom X F.left
    k : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1)
    i : Nat
    hi : LT.lt i (HAdd.hAdd (HAdd.hAdd n 1) 1)
    j : Nat
    hj : LT.lt j (HAdd.hAdd (HAdd.hAdd n 1) 1)
    hjk : LE.le ⟨j, hj⟩ k
    hij : LE.le ⟨i, hi⟩ ⟨j, hj⟩
    ⊢ Eq (CategoryTheory.ComposableArrows.Precomp.map F f ⟨i, hi⟩ k ⋯) (CategoryTh …
  -/
  obtain ⟨k, hk⟩ := k
  /-
    case mk.mk.mk
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    n : Nat
    F : CategoryTheory.ComposableArrows C n
    X : C
    f : Quiver.Hom X F.left
    i : Nat
    hi : LT.lt i (HAdd.hAdd (HAdd.hAdd n 1) 1)
    j : Nat
    hj : LT.lt j (HAdd.hAdd (HAdd.hAdd n 1) 1)
    hij : LE.le ⟨i, hi⟩ ⟨j, hj⟩
    k : Nat
    hk : LT.lt k (HAdd.hAdd (HAdd.hAdd n 1) 1)
    hjk : LE.le ⟨j, hj⟩ ⟨k, hk⟩
    ⊢ Eq (CategoryTheory.ComposableArrows.Precomp.map F f ⟨i, hi⟩ ⟨k, hk⟩ ⋯) (Cate …
  -/
  cases i
    /-
      case mk.mk.mk.zero
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      n : Nat
      F : CategoryTheory.ComposableArrows C n
      X : C
      f : Quiver.Hom X F.left
      j : Nat
      hj : LT.lt j (HAdd.hAdd (HAdd.hAdd n 1) 1)
      k : Nat
      hk : LT.lt k (HAdd.hAdd (HAdd.hAdd n 1) 1)
      hjk : LE.le ⟨j, hj⟩ ⟨k, hk⟩
      hi : LT.lt 0 (HAdd.hAdd (HAdd.hAdd n 1) 1)
      hij : LE.le ⟨0, hi⟩ ⟨j, hj⟩
      ⊢ Eq (CategoryTheory.ComposableArrows.Precomp.map F f ⟨0, hi⟩ ⟨k, hk⟩ ⋯) (Cate …
    -/
  · obtain _ | _ | j := j
      /-
        case mk.mk.mk.zero.zero
        C : Type u_1
        inst✝ : CategoryTheory.Category.{u_2, u_1} C
        n : Nat
        F : CategoryTheory.ComposableArrows C n
        X : C
        f : Quiver.Hom X F.left
        k : Nat
        hk : LT.lt k (HAdd.hAdd (HAdd.hAdd n 1) 1)
        hi hj : LT.lt 0 (HAdd.hAdd (HAdd.hAdd n 1) 1)
        hjk : LE.le ⟨0, hj⟩ ⟨k, hk⟩
        hij : LE.le ⟨0, hi⟩ ⟨0, hj⟩
        ⊢ Eq (CategoryTheory.ComposableArrows.Precomp.map F f ⟨0, hi⟩ ⟨k, hk⟩ ⋯) (Cate …
      -/
    · dsimp
      /-
        case mk.mk.mk.zero.zero
        C : Type u_1
        inst✝ : CategoryTheory.Category.{u_2, u_1} C
        n : Nat
        F : CategoryTheory.ComposableArrows C n
        X : C
        f : Quiver.Hom X F.left
        k : Nat
        hk : LT.lt k (HAdd.hAdd (HAdd.hAdd n 1) 1)
        hi hj : LT.lt 0 (HAdd.hAdd (HAdd.hAdd n 1) 1)
        hjk : LE.le ⟨0, hj⟩ ⟨k, hk⟩
        hij : LE.le ⟨0, hi⟩ ⟨0, hj⟩
        ⊢ Eq (CategoryTheory.ComposableArrows.Precomp.map F f 0 ⟨k, hk⟩ ⋯) (CategoryTh …
      -/
      rw [id_comp]
      /-
        🎉 no goals
      -/
      /-
        case mk.mk.mk.zero.succ.zero
        C : Type u_1
        inst✝ : CategoryTheory.Category.{u_2, u_1} C
        n : Nat
        F : CategoryTheory.ComposableArrows C n
        X : C
        f : Quiver.Hom X F.left
        k : Nat
        hk : LT.lt k (HAdd.hAdd (HAdd.hAdd n 1) 1)
        hi : LT.lt 0 (HAdd.hAdd (HAdd.hAdd n 1) 1)
        hj : LT.lt (HAdd.hAdd 0 1) (HAdd.hAdd (HAdd.hAdd n 1) 1)
        hjk : LE.le ⟨HAdd.hAdd 0 1, hj⟩ ⟨k, hk⟩
        hij : LE.le ⟨0, hi⟩ ⟨HAdd.hAdd 0 1, hj⟩
        ⊢ Eq (CategoryTheory.ComposableArrows.Precomp.map F f ⟨0, hi⟩ ⟨k, hk⟩ ⋯) (Cate …
      -/
    · obtain _ | _ | k := k
        /-
          case mk.mk.mk.zero.succ.zero.zero
          C : Type u_1
          inst✝ : CategoryTheory.Category.{u_2, u_1} C
          n : Nat
          F : CategoryTheory.ComposableArrows C n
          X : C
          f : Quiver.Hom X F.left
          hi : LT.lt 0 (HAdd.hAdd (HAdd.hAdd n 1) 1)
          hj : LT.lt (HAdd.hAdd 0 1) (HAdd.hAdd (HAdd.hAdd n 1) 1)
          hij : LE.le ⟨0, hi⟩ ⟨HAdd.hAdd 0 1, hj⟩
          hk : LT.lt 0 (HAdd.hAdd (HAdd.hAdd n 1) 1)
          hjk : LE.le ⟨HAdd.hAdd 0 1, hj⟩ ⟨0, hk⟩
          ⊢ Eq (CategoryTheory.ComposableArrows.Precomp.map F f ⟨0, hi⟩ ⟨0, hk⟩ ⋯) (Cate …
        -/
      · simp [Nat.succ.injEq] at hjk
        /-
          🎉 no goals
        -/
        /-
          case mk.mk.mk.zero.succ.zero.succ.zero
          C : Type u_1
          inst✝ : CategoryTheory.Category.{u_2, u_1} C
          n : Nat
          F : CategoryTheory.ComposableArrows C n
          X : C
          f : Quiver.Hom X F.left
          hi : LT.lt 0 (HAdd.hAdd (HAdd.hAdd n 1) 1)
          hj : LT.lt (HAdd.hAdd 0 1) (HAdd.hAdd (HAdd.hAdd n 1) 1)
          hij : LE.le ⟨0, hi⟩ ⟨HAdd.hAdd 0 1, hj⟩
          hk : LT.lt (HAdd.hAdd 0 1) (HAdd.hAdd (HAdd.hAdd n 1) 1)
          hjk : LE.le ⟨HAdd.hAdd 0 1, hj⟩ ⟨HAdd.hAdd 0 1, hk⟩
          ⊢ Eq (CategoryTheory.ComposableArrows.Precomp.map F f ⟨0, hi⟩ ⟨HAdd.hAdd 0 1,  …
        -/
      · simp
        /-
          🎉 no goals
        -/
        /-
          case mk.mk.mk.zero.succ.zero.succ.succ
          C : Type u_1
          inst✝ : CategoryTheory.Category.{u_2, u_1} C
          n : Nat
          F : CategoryTheory.ComposableArrows C n
          X : C
          f : Quiver.Hom X F.left
          hi : LT.lt 0 (HAdd.hAdd (HAdd.hAdd n 1) 1)
          hj : LT.lt (HAdd.hAdd 0 1) (HAdd.hAdd (HAdd.hAdd n 1) 1)
          hij : LE.le ⟨0, hi⟩ ⟨HAdd.hAdd 0 1, hj⟩
          k : Nat
          hk : LT.lt (HAdd.hAdd (HAdd.hAdd k 1) 1) (HAdd.hAdd (HAdd.hAdd n 1) 1)
          hjk : LE.le ⟨HAdd.hAdd 0 1, hj⟩ ⟨HAdd.hAdd (HAdd.hAdd k 1) 1, hk⟩
          ⊢ Eq (CategoryTheory.ComposableArrows.Precomp.map F f ⟨0, hi⟩ ⟨HAdd.hAdd (HAdd …
        -/
      · rfl
        /-
          🎉 no goals
        -/
      /-
        case mk.mk.mk.zero.succ.succ
        C : Type u_1
        inst✝ : CategoryTheory.Category.{u_2, u_1} C
        n : Nat
        F : CategoryTheory.ComposableArrows C n
        X : C
        f : Quiver.Hom X F.left
        k : Nat
        hk : LT.lt k (HAdd.hAdd (HAdd.hAdd n 1) 1)
        hi : LT.lt 0 (HAdd.hAdd (HAdd.hAdd n 1) 1)
        j : Nat
        hj : LT.lt (HAdd.hAdd (HAdd.hAdd j 1) 1) (HAdd.hAdd (HAdd.hAdd n 1) 1)
        hjk : LE.le ⟨HAdd.hAdd (HAdd.hAdd j 1) 1, hj⟩ ⟨k, hk⟩
        hij : LE.le ⟨0, hi⟩ ⟨HAdd.hAdd (HAdd.hAdd j 1) 1, hj⟩
        ⊢ Eq (CategoryTheory.ComposableArrows.Precomp.map F f ⟨0, hi⟩ ⟨k, hk⟩ ⋯) (Cate …
      -/
    · obtain _ | _ | k := k
        /-
          case mk.mk.mk.zero.succ.succ.zero
          C : Type u_1
          inst✝ : CategoryTheory.Category.{u_2, u_1} C
          n : Nat
          F : CategoryTheory.ComposableArrows C n
          X : C
          f : Quiver.Hom X F.left
          hi : LT.lt 0 (HAdd.hAdd (HAdd.hAdd n 1) 1)
          j : Nat
          hj : LT.lt (HAdd.hAdd (HAdd.hAdd j 1) 1) (HAdd.hAdd (HAdd.hAdd n 1) 1)
          hij : LE.le ⟨0, hi⟩ ⟨HAdd.hAdd (HAdd.hAdd j 1) 1, hj⟩
          hk : LT.lt 0 (HAdd.hAdd (HAdd.hAdd n 1) 1)
          hjk : LE.le ⟨HAdd.hAdd (HAdd.hAdd j 1) 1, hj⟩ ⟨0, hk⟩
          ⊢ Eq (CategoryTheory.ComposableArrows.Precomp.map F f ⟨0, hi⟩ ⟨0, hk⟩ ⋯) (Cate …
        -/
      · simp [Fin.ext_iff] at hjk
        /-
          🎉 no goals
        -/
        /-
          case mk.mk.mk.zero.succ.succ.succ.zero
          C : Type u_1
          inst✝ : CategoryTheory.Category.{u_2, u_1} C
          n : Nat
          F : CategoryTheory.ComposableArrows C n
          X : C
          f : Quiver.Hom X F.left
          hi : LT.lt 0 (HAdd.hAdd (HAdd.hAdd n 1) 1)
          j : Nat
          hj : LT.lt (HAdd.hAdd (HAdd.hAdd j 1) 1) (HAdd.hAdd (HAdd.hAdd n 1) 1)
          hij : LE.le ⟨0, hi⟩ ⟨HAdd.hAdd (HAdd.hAdd j 1) 1, hj⟩
          hk : LT.lt (HAdd.hAdd 0 1) (HAdd.hAdd (HAdd.hAdd n 1) 1)
          hjk : LE.le ⟨HAdd.hAdd (HAdd.hAdd j 1) 1, hj⟩ ⟨HAdd.hAdd 0 1, hk⟩
          ⊢ Eq (CategoryTheory.ComposableArrows.Precomp.map F f ⟨0, hi⟩ ⟨HAdd.hAdd 0 1,  …
        -/
      · simp [Fin.le_def] at hjk
        /-
          case mk.mk.mk.zero.succ.succ.succ.zero
          C : Type u_1
          inst✝ : CategoryTheory.Category.{u_2, u_1} C
          n : Nat
          F : CategoryTheory.ComposableArrows C n
          X : C
          f : Quiver.Hom X F.left
          hi : LT.lt 0 (HAdd.hAdd (HAdd.hAdd n 1) 1)
          j : Nat
          hj : LT.lt (HAdd.hAdd (HAdd.hAdd j 1) 1) (HAdd.hAdd (HAdd.hAdd n 1) 1)
          hij : LE.le ⟨0, hi⟩ ⟨HAdd.hAdd (HAdd.hAdd j 1) 1, hj⟩
          hk : LT.lt (HAdd.hAdd 0 1) (HAdd.hAdd (HAdd.hAdd n 1) 1)
          hjk✝ : LE.le ⟨HAdd.hAdd (HAdd.hAdd j 1) 1, hj⟩ ⟨HAdd.hAdd 0 1, hk⟩
          hjk : LE.le (HAdd.hAdd (HAdd.hAdd j 1) 1) 1
          ⊢ Eq (CategoryTheory.ComposableArrows.Precomp.map F f ⟨0, hi⟩ ⟨HAdd.hAdd 0 1,  …
        -/
        omega
        /-
          🎉 no goals
        -/
        /-
          case mk.mk.mk.zero.succ.succ.succ.succ
          C : Type u_1
          inst✝ : CategoryTheory.Category.{u_2, u_1} C
          n : Nat
          F : CategoryTheory.ComposableArrows C n
          X : C
          f : Quiver.Hom X F.left
          hi : LT.lt 0 (HAdd.hAdd (HAdd.hAdd n 1) 1)
          j : Nat
          hj : LT.lt (HAdd.hAdd (HAdd.hAdd j 1) 1) (HAdd.hAdd (HAdd.hAdd n 1) 1)
          hij : LE.le ⟨0, hi⟩ ⟨HAdd.hAdd (HAdd.hAdd j 1) 1, hj⟩
          k : Nat
          hk : LT.lt (HAdd.hAdd (HAdd.hAdd k 1) 1) (HAdd.hAdd (HAdd.hAdd n 1) 1)
          hjk : LE.le ⟨HAdd.hAdd (HAdd.hAdd j 1) 1, hj⟩ ⟨HAdd.hAdd (HAdd.hAdd k 1) 1, hk⟩
          ⊢ Eq (CategoryTheory.ComposableArrows.Precomp.map F f ⟨0, hi⟩ ⟨HAdd.hAdd (HAdd …
        -/
      · dsimp
        /-
          case mk.mk.mk.zero.succ.succ.succ.succ
          C : Type u_1
          inst✝ : CategoryTheory.Category.{u_2, u_1} C
          n : Nat
          F : CategoryTheory.ComposableArrows C n
          X : C
          f : Quiver.Hom X F.left
          hi : LT.lt 0 (HAdd.hAdd (HAdd.hAdd n 1) 1)
          j : Nat
          hj : LT.lt (HAdd.hAdd (HAdd.hAdd j 1) 1) (HAdd.hAdd (HAdd.hAdd n 1) 1)
          hij : LE.le ⟨0, hi⟩ ⟨HAdd.hAdd (HAdd.hAdd j 1) 1, hj⟩
          k : Nat
          hk : LT.lt (HAdd.hAdd (HAdd.hAdd k 1) 1) (HAdd.hAdd (HAdd.hAdd n 1) 1)
          hjk : LE.le ⟨HAdd.hAdd (HAdd.hAdd j 1) 1, hj⟩ ⟨HAdd.hAdd (HAdd.hAdd k 1) 1, hk⟩
          ⊢ Eq (CategoryTheory.CategoryStruct.comp f (F.map (CategoryTheory.homOfLE ⋯))) …
        -/
        rw [assoc, ← F.map_comp, homOfLE_comp]
        /-
          🎉 no goals
        -/
    /-
      case mk.mk.mk.succ
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      n : Nat
      F : CategoryTheory.ComposableArrows C n
      X : C
      f : Quiver.Hom X F.left
      j : Nat
      hj : LT.lt j (HAdd.hAdd (HAdd.hAdd n 1) 1)
      k : Nat
      hk : LT.lt k (HAdd.hAdd (HAdd.hAdd n 1) 1)
      hjk : LE.le ⟨j, hj⟩ ⟨k, hk⟩
      n✝ : Nat
      hi : LT.lt (HAdd.hAdd n✝ 1) (HAdd.hAdd (HAdd.hAdd n 1) 1)
      hij : LE.le ⟨HAdd.hAdd n✝ 1, hi⟩ ⟨j, hj⟩
      ⊢ Eq (CategoryTheory.ComposableArrows.Precomp.map F f ⟨HAdd.hAdd n✝ 1, hi⟩ ⟨k, …
    -/
  · obtain _ | j := j
      /-
        case mk.mk.mk.succ.zero
        C : Type u_1
        inst✝ : CategoryTheory.Category.{u_2, u_1} C
        n : Nat
        F : CategoryTheory.ComposableArrows C n
        X : C
        f : Quiver.Hom X F.left
        k : Nat
        hk : LT.lt k (HAdd.hAdd (HAdd.hAdd n 1) 1)
        n✝ : Nat
        hi : LT.lt (HAdd.hAdd n✝ 1) (HAdd.hAdd (HAdd.hAdd n 1) 1)
        hj : LT.lt 0 (HAdd.hAdd (HAdd.hAdd n 1) 1)
        hjk : LE.le ⟨0, hj⟩ ⟨k, hk⟩
        hij : LE.le ⟨HAdd.hAdd n✝ 1, hi⟩ ⟨0, hj⟩
        ⊢ Eq (CategoryTheory.ComposableArrows.Precomp.map F f ⟨HAdd.hAdd n✝ 1, hi⟩ ⟨k, …
      -/
    · simp [Fin.ext_iff] at hij
      /-
        🎉 no goals
      -/
      /-
        case mk.mk.mk.succ.succ
        C : Type u_1
        inst✝ : CategoryTheory.Category.{u_2, u_1} C
        n : Nat
        F : CategoryTheory.ComposableArrows C n
        X : C
        f : Quiver.Hom X F.left
        k : Nat
        hk : LT.lt k (HAdd.hAdd (HAdd.hAdd n 1) 1)
        n✝ : Nat
        hi : LT.lt (HAdd.hAdd n✝ 1) (HAdd.hAdd (HAdd.hAdd n 1) 1)
        j : Nat
        hj : LT.lt (HAdd.hAdd j 1) (HAdd.hAdd (HAdd.hAdd n 1) 1)
        hjk : LE.le ⟨HAdd.hAdd j 1, hj⟩ ⟨k, hk⟩
        hij : LE.le ⟨HAdd.hAdd n✝ 1, hi⟩ ⟨HAdd.hAdd j 1, hj⟩
        ⊢ Eq (CategoryTheory.ComposableArrows.Precomp.map F f ⟨HAdd.hAdd n✝ 1, hi⟩ ⟨k, …
      -/
    · obtain _ | k := k
        /-
          case mk.mk.mk.succ.succ.zero
          C : Type u_1
          inst✝ : CategoryTheory.Category.{u_2, u_1} C
          n : Nat
          F : CategoryTheory.ComposableArrows C n
          X : C
          f : Quiver.Hom X F.left
          n✝ : Nat
          hi : LT.lt (HAdd.hAdd n✝ 1) (HAdd.hAdd (HAdd.hAdd n 1) 1)
          j : Nat
          hj : LT.lt (HAdd.hAdd j 1) (HAdd.hAdd (HAdd.hAdd n 1) 1)
          hij : LE.le ⟨HAdd.hAdd n✝ 1, hi⟩ ⟨HAdd.hAdd j 1, hj⟩
          hk : LT.lt 0 (HAdd.hAdd (HAdd.hAdd n 1) 1)
          hjk : LE.le ⟨HAdd.hAdd j 1, hj⟩ ⟨0, hk⟩
          ⊢ Eq (CategoryTheory.ComposableArrows.Precomp.map F f ⟨HAdd.hAdd n✝ 1, hi⟩ ⟨0, …
        -/
      · simp [Fin.ext_iff] at hjk
        /-
          🎉 no goals
        -/
        /-
          case mk.mk.mk.succ.succ.succ
          C : Type u_1
          inst✝ : CategoryTheory.Category.{u_2, u_1} C
          n : Nat
          F : CategoryTheory.ComposableArrows C n
          X : C
          f : Quiver.Hom X F.left
          n✝ : Nat
          hi : LT.lt (HAdd.hAdd n✝ 1) (HAdd.hAdd (HAdd.hAdd n 1) 1)
          j : Nat
          hj : LT.lt (HAdd.hAdd j 1) (HAdd.hAdd (HAdd.hAdd n 1) 1)
          hij : LE.le ⟨HAdd.hAdd n✝ 1, hi⟩ ⟨HAdd.hAdd j 1, hj⟩
          k : Nat
          hk : LT.lt (HAdd.hAdd k 1) (HAdd.hAdd (HAdd.hAdd n 1) 1)
          hjk : LE.le ⟨HAdd.hAdd j 1, hj⟩ ⟨HAdd.hAdd k 1, hk⟩
          ⊢ Eq (CategoryTheory.ComposableArrows.Precomp.map F f ⟨HAdd.hAdd n✝ 1, hi⟩ ⟨HA …
        -/
      · dsimp
        /-
          case mk.mk.mk.succ.succ.succ
          C : Type u_1
          inst✝ : CategoryTheory.Category.{u_2, u_1} C
          n : Nat
          F : CategoryTheory.ComposableArrows C n
          X : C
          f : Quiver.Hom X F.left
          n✝ : Nat
          hi : LT.lt (HAdd.hAdd n✝ 1) (HAdd.hAdd (HAdd.hAdd n 1) 1)
          j : Nat
          hj : LT.lt (HAdd.hAdd j 1) (HAdd.hAdd (HAdd.hAdd n 1) 1)
          hij : LE.le ⟨HAdd.hAdd n✝ 1, hi⟩ ⟨HAdd.hAdd j 1, hj⟩
          k : Nat
          hk : LT.lt (HAdd.hAdd k 1) (HAdd.hAdd (HAdd.hAdd n 1) 1)
          hjk : LE.le ⟨HAdd.hAdd j 1, hj⟩ ⟨HAdd.hAdd k 1, hk⟩
          ⊢ Eq (F.map (CategoryTheory.homOfLE ⋯)) (CategoryTheory.CategoryStruct.comp (F …
        -/
        rw [← F.map_comp, homOfLE_comp]
        /-
          🎉 no goals
        -/


/-- "Precomposition" of `F : ComposableArrows C n` by a morphism `f : X ⟶ F.left`. -/
@[simps]
def precomp {X : C} (f : X ⟶ F.left) : ComposableArrows C (n + 1) where
  obj := Precomp.obj F X
  map g := Precomp.map F f _ _ (leOfHom g)
  map_id := Precomp.map_id F f
  map_comp g g' := Precomp.map_comp F f (leOfHom g) (leOfHom g')


/-- Constructor for `ComposableArrows C 2`. -/
@[simp]
def mk₂ {X₀ X₁ X₂ : C} (f : X₀ ⟶ X₁) (g : X₁ ⟶ X₂) : ComposableArrows C 2 :=
  (mk₁ g).precomp f


/-- Constructor for `ComposableArrows C 3`. -/
@[simp]
def mk₃ {X₀ X₁ X₂ X₃ : C} (f : X₀ ⟶ X₁) (g : X₁ ⟶ X₂) (h : X₂ ⟶ X₃) : ComposableArrows C 3 :=
  (mk₂ g h).precomp f


/-- Constructor for `ComposableArrows C 4`. -/
@[simp]
def mk₄ {X₀ X₁ X₂ X₃ X₄ : C} (f : X₀ ⟶ X₁) (g : X₁ ⟶ X₂) (h : X₂ ⟶ X₃) (i : X₃ ⟶ X₄) :
    ComposableArrows C 4 :=
  (mk₃ g h i).precomp f


/-- Constructor for `ComposableArrows C 5`. -/
@[simp]
def mk₅ {X₀ X₁ X₂ X₃ X₄ X₅ : C} (f : X₀ ⟶ X₁) (g : X₁ ⟶ X₂) (h : X₂ ⟶ X₃)
    (i : X₃ ⟶ X₄) (j : X₄ ⟶ X₅) :
    ComposableArrows C 5 :=
  (mk₄ g h i j).precomp f


/-- The map `ComposableArrows C m → ComposableArrows C n` obtained by precomposition with
a functor `Fin (n + 1) ⥤ Fin (m + 1)`. -/
@[simps!]
def whiskerLeft (F : ComposableArrows C m) (Φ : Fin (n + 1) ⥤ Fin (m + 1)) :
    ComposableArrows C n := Φ ⋙ F


/-- The functor `ComposableArrows C m ⥤ ComposableArrows C n` obtained by precomposition with
a functor `Fin (n + 1) ⥤ Fin (m + 1)`. -/
@[simps!]
def whiskerLeftFunctor (Φ : Fin (n + 1) ⥤ Fin (m + 1)) :
    ComposableArrows C m ⥤ ComposableArrows C n where
  obj F := F.whiskerLeft Φ
  map f := CategoryTheory.whiskerLeft Φ f


/-- The functor `Fin n ⥤ Fin (n + 1)` which sends `i` to `i.succ`. -/
@[simps]
def _root_.Fin.succFunctor (n : ℕ) : Fin n ⥤ Fin (n + 1) where
  obj i := i.succ
  map {_ _} hij := homOfLE (Fin.succ_le_succ_iff.2 (leOfHom hij))


/-- The functor `ComposableArrows C (n + 1) ⥤ ComposableArrows C n` which forgets
the first arrow. -/
@[simps!]
def δ₀Functor : ComposableArrows C (n + 1) ⥤ ComposableArrows C n :=
  whiskerLeftFunctor (Fin.succFunctor (n + 1))


/-- The `ComposableArrows C n` obtained by forgetting the first arrow. -/
abbrev δ₀ (F : ComposableArrows C (n + 1)) := δ₀Functor.obj F


@[simp]
lemma precomp_δ₀ {X : C} (f : X ⟶ F.left) : (F.precomp f).δ₀ = F := rfl


/-- The functor `Fin n ⥤ Fin (n + 1)` which sends `i` to `i.castSucc`. -/
@[simps]
def _root_.Fin.castSuccFunctor (n : ℕ) : Fin n ⥤ Fin (n + 1) where
  obj i := i.castSucc
  map hij := hij


/-- The functor `ComposableArrows C (n + 1) ⥤ ComposableArrows C n` which forgets
the last arrow. -/
@[simps!]
def δlastFunctor : ComposableArrows C (n + 1) ⥤ ComposableArrows C n :=
  whiskerLeftFunctor (Fin.castSuccFunctor (n + 1))


/-- The `ComposableArrows C n` obtained by forgetting the first arrow. -/
abbrev δlast (F : ComposableArrows C (n + 1)) := δlastFunctor.obj F


/-- Inductive construction of morphisms in `ComposableArrows C (n + 1)`: in order to construct
a morphism `F ⟶ G`, it suffices to provide `α : F.obj' 0 ⟶ G.obj' 0` and `β : F.δ₀ ⟶ G.δ₀`
such that `F.map' 0 1 ≫ app' β 0 = α ≫ G.map' 0 1`. -/
                   /-
                     C : Type u_1
                     inst✝ : CategoryTheory.Category.{?u.145859, u_1} C
                     n m : Nat
                     F✝ G✝ : CategoryTheory.ComposableArrows C n
                     F G : CategoryTheory.ComposableArrows C (HAdd.hAdd n 1)
                     ⊢ LE.le 0 (HAdd.hAdd n 1)
                   -/
                   /-
                     🎉 no goals
                   -/
def homMkSucc (α : F.obj' 0 ⟶ G.obj' 0) (β : F.δ₀ ⟶ G.δ₀)
                              /-
                                🎉 no goals
                              -/
         /-
           C : Type u_1
           inst✝ : CategoryTheory.Category.{?u.145859, u_1} C
           n m : Nat
           F✝ G✝ : CategoryTheory.ComposableArrows C n
           F G : CategoryTheory.ComposableArrows C (HAdd.hAdd n 1)
           α : Quiver.Hom (F.obj' 0 ⋯) (G.obj' 0 ⋯)
           β : Quiver.Hom F.δ₀ G.δ₀
           ⊢ LE.le 0 1
         -/
         /-
           🎉 no goals
         -/
         /-
           🎉 no goals
         -/
                      /-
                        🎉 no goals
                      -/
                                     /-
                                       🎉 no goals
                                     -/
    (w : F.map' 0 1 ≫ app' β 0 = α ≫ G.map' 0 1) : F ⟶ G :=
                                     /-
                                       🎉 no goals
                                     -/
  homMk
    (fun i => match i with
      | ⟨0, _⟩ => α
                       /-
                         C : Type u_1
                         inst✝ : CategoryTheory.Category.{?u.145859, u_1} C
                         n m : Nat
                         F✝ G✝ : CategoryTheory.ComposableArrows C n
                         F G : CategoryTheory.ComposableArrows C (HAdd.hAdd n 1)
                         α : Quiver.Hom (F.obj' 0 ⋯) (G.obj' 0 ⋯)
                         β : Quiver.Hom F.δ₀ G.δ₀
                         w : Eq (CategoryTheory.CategoryStruct.comp (F.map' 0 1 ⋯ ⋯) (CategoryTheory.Co …
                         i✝ : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1)
                         i : Nat
                         hi : LT.lt (HAdd.hAdd i 1) (HAdd.hAdd (HAdd.hAdd n 1) 1)
                         ⊢ LE.le i n
                       -/
      | ⟨i + 1, hi⟩ => app' β i)
                       /-
                         🎉 no goals
                       -/
    (fun i hi => by
      /-
        C : Type u_1
        inst✝ : CategoryTheory.Category.{?u.145859, u_1} C
        n m : Nat
        F✝ G✝ : CategoryTheory.ComposableArrows C n
        F G : CategoryTheory.ComposableArrows C (HAdd.hAdd n 1)
        α : Quiver.Hom (F.obj' 0 ⋯) (G.obj' 0 ⋯)
        β : Quiver.Hom F.δ₀ G.δ₀
        w : Eq (CategoryTheory.CategoryStruct.comp (F.map' 0 1 ⋯ ⋯) (CategoryTheory.Co …
        i : Nat
        hi : LT.lt i (HAdd.hAdd n 1)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map' i (HAdd.hAdd i 1) ⋯ hi) ((fun …
      -/
      obtain _ | i := i
        /-
          case zero
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.145859, u_1} C
          n m : Nat
          F✝ G✝ : CategoryTheory.ComposableArrows C n
          F G : CategoryTheory.ComposableArrows C (HAdd.hAdd n 1)
          α : Quiver.Hom (F.obj' 0 ⋯) (G.obj' 0 ⋯)
          β : Quiver.Hom F.δ₀ G.δ₀
          w : Eq (CategoryTheory.CategoryStruct.comp (F.map' 0 1 ⋯ ⋯) (CategoryTheory.Co …
          hi : LT.lt 0 (HAdd.hAdd n 1)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map' 0 (HAdd.hAdd 0 1) ⋯ hi) ((fun …
        -/
      · exact w
        /-
          🎉 no goals
        -/
        /-
          case succ
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.145859, u_1} C
          n m : Nat
          F✝ G✝ : CategoryTheory.ComposableArrows C n
          F G : CategoryTheory.ComposableArrows C (HAdd.hAdd n 1)
          α : Quiver.Hom (F.obj' 0 ⋯) (G.obj' 0 ⋯)
          β : Quiver.Hom F.δ₀ G.δ₀
          w : Eq (CategoryTheory.CategoryStruct.comp (F.map' 0 1 ⋯ ⋯) (CategoryTheory.Co …
          i : Nat
          hi : LT.lt (HAdd.hAdd i 1) (HAdd.hAdd n 1)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map' (HAdd.hAdd i 1) (HAdd.hAdd (H …
        -/
      · exact naturality' β i (i + 1))
        /-
          🎉 no goals
        -/


            n m : Nat
                F✝ G✝ : CategoryTheory.ComposableArrows C n
                F G : CategoryTheory.ComposableArrows C (HAdd.hAdd n 1)
                ⊢ LE.le 0 (HAdd.hAdd n 1)
              -/
              /-
                🎉 no goals
              -/
variable (α : F.obj' 0 ⟶ G.obj' 0) (β : F.δ₀ ⟶ G.δ₀)
                         /-
                           🎉 no goals
                         -/
       /-
         C : Type u_1
         inst✝ : CategoryTheory.Category.{?u.152057, u_1} C
         n m : Nat
         F✝ G✝ : CategoryTheory.ComposableArrows C n
         F G : CategoryTheory.ComposableArrows C (HAdd.hAdd n 1)
         α : Quiver.Hom (F.obj' 0 ⋯) (G.obj' 0 ⋯)
         β : Quiver.Hom F.δ₀ G.δ₀
         ⊢ LE.le 0 1
       -/
       /-
         🎉 no goals
       -/
       /-
         🎉 no goals
       -/
                    /-
                      🎉 no goals
                    -/
                                   /-
                                     🎉 no goals
                                   -/
  (w : F.map' 0 1 ≫ app' β 0 = α ≫ G.map' 0 1)
                                   /-
                                     🎉 no goals
                                   -/

@[simp]
lemma homMkSucc_app_zero : (homMkSucc α β w).app 0 = α := rfl


eArrows C n
                F G : CategoryTheory.ComposableArrows C (HAdd.hAdd n 1)
                ⊢ LE.le 0 (HAdd.hAdd n 1)
              -/
              /-
                🎉 no goals
              -/
variable (α : F.obj' 0 ⟶ G.obj' 0) (β : F.δ₀ ⟶ G.δ₀)
                         /-
                           🎉 no goals
                         -/
       /-
         C : Type u_1
         inst✝ : CategoryTheory.Category.{?u.153755, u_1} C
         n m : Nat
         F✝ G✝ : CategoryTheory.ComposableArrows C n
         F G : CategoryTheory.ComposableArrows C (HAdd.hAdd n 1)
         α : Quiver.Hom (F.obj' 0 ⋯) (G.obj' 0 ⋯)
         β : Quiver.Hom F.δ₀ G.δ₀
         ⊢ LE.le 0 1
       -/
       /-
         🎉 no goals
       -/
       /-
         🎉 no goals
       -/
                    /-
                      🎉 no goals
                    -/
                                   /-
                                     🎉 no goals
                                   -/
  (w : F.map' 0 1 ≫ app' β 0 = α ≫ G.map' 0 1)
                                   /-
                                     🎉 no goals
                                   -/

@[simp]
lemma homMkSucc_app_zero : (homMkSucc α β w).app 0 = α := rfl

@[simp]
lemma homMkSucc_app_succ (i : ℕ) (hi : i + 1 < n + 1 + 1) :
                                        /-
                                          C : Type u_1
                                          inst✝ : CategoryTheory.Category.{?u.153755, u_1} C
                                          n m : Nat
                                          F✝ G✝ : CategoryTheory.ComposableArrows C n
                                          F G : CategoryTheory.ComposableArrows C (HAdd.hAdd n 1)
                                          α : Quiver.Hom (F.obj' 0 ⋯) (G.obj' 0 ⋯)
                                          β : Quiver.Hom F.δ₀ G.δ₀
                                          w : Eq (CategoryTheory.CategoryStruct.comp (F.map' 0 1 ⋯ ⋯) (CategoryTheory.Co …
                                          i : Nat
                                          hi : LT.lt (HAdd.hAdd i 1) (HAdd.hAdd (HAdd.hAdd n 1) 1)
                                          ⊢ LE.le i n
                                        -/
    (homMkSucc α β w).app ⟨i + 1, hi⟩ = app' β i := rfl
                                        /-
                                          🎉 no goals
                                        -/


lemma hom_ext_succ {F G : ComposableArrows C (n + 1)} {f g : F ⟶ G}
          /-
            C : Type u_1
            inst✝ : CategoryTheory.Category.{?u.155827, u_1} C
            n m : Nat
            F✝ G✝ : CategoryTheory.ComposableArrows C n
            F G : CategoryTheory.ComposableArrows C (HAdd.hAdd n 1)
            f g : Quiver.Hom F G
            ⊢ LE.le 0 (HAdd.hAdd n 1)
          -/
          /-
            🎉 no goals
          -/
    (h₀ : app' f 0 = app' g 0) (h₁ : δ₀Functor.map f = δ₀Functor.map g) : f = g := by
                     /-
                       🎉 no goals
                     -/
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    n : Nat
    F G : CategoryTheory.ComposableArrows C (HAdd.hAdd n 1)
    f g : Quiver.Hom F G
    h₀ : Eq (CategoryTheory.ComposableArrows.app' f 0 ⋯) (CategoryTheory.Composabl …
    h₁ : Eq (CategoryTheory.ComposableArrows.δ₀Functor.map f) (CategoryTheory.Comp …
    ⊢ Eq f g
  -/
  ext ⟨i, hi⟩
  /-
    case w.h.mk
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    n : Nat
    F G : CategoryTheory.ComposableArrows C (HAdd.hAdd n 1)
    f g : Quiver.Hom F G
    h₀ : Eq (CategoryTheory.ComposableArrows.app' f 0 ⋯) (CategoryTheory.Composabl …
    h₁ : Eq (CategoryTheory.ComposableArrows.δ₀Functor.map f) (CategoryTheory.Comp …
    i : Nat
    hi : LT.lt i (HAdd.hAdd (HAdd.hAdd n 1) 1)
    ⊢ Eq (f.app ⟨i, hi⟩) (g.app ⟨i, hi⟩)
  -/
  obtain _ | i := i
    /-
      case w.h.mk.zero
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      n : Nat
      F G : CategoryTheory.ComposableArrows C (HAdd.hAdd n 1)
      f g : Quiver.Hom F G
      h₀ : Eq (CategoryTheory.ComposableArrows.app' f 0 ⋯) (CategoryTheory.Composabl …
      h₁ : Eq (CategoryTheory.ComposableArrows.δ₀Functor.map f) (CategoryTheory.Comp …
      hi : LT.lt 0 (HAdd.hAdd (HAdd.hAdd n 1) 1)
      ⊢ Eq (f.app ⟨0, hi⟩) (g.app ⟨0, hi⟩)
    -/
  · exact h₀
    /-
      🎉 no goals
    -/
    /-
      case w.h.mk.succ
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      n : Nat
      F G : CategoryTheory.ComposableArrows C (HAdd.hAdd n 1)
      f g : Quiver.Hom F G
      h₀ : Eq (CategoryTheory.ComposableArrows.app' f 0 ⋯) (CategoryTheory.Composabl …
      h₁ : Eq (CategoryTheory.ComposableArrows.δ₀Functor.map f) (CategoryTheory.Comp …
      i : Nat
      hi : LT.lt (HAdd.hAdd i 1) (HAdd.hAdd (HAdd.hAdd n 1) 1)
      ⊢ Eq (f.app ⟨HAdd.hAdd i 1, hi⟩) (g.app ⟨HAdd.hAdd i 1, hi⟩)
    -/
  · exact congr_app h₁ ⟨i, by valid⟩
    /-
      🎉 no goals
    -/


/-- Inductive construction of isomorphisms in `ComposableArrows C (n + 1)`: in order to
construct an isomorphism `F ≅ G`, it suffices to provide `α : F.obj' 0 ≅ G.obj' 0` and
`β : F.δ₀ ≅ G.δ₀` such that `F.map' 0 1 ≫ app' β.hom 0 = α.hom ≫ G.map' 0 1`. -/
@[simps]
                                                      /-
                                                        C : Type u_1
                                                        inst✝ : CategoryTheory.Category.{?u.157328, u_1} C
                                                        n m : Nat
                                                        F✝ G✝ : CategoryTheory.ComposableArrows C n
                                                        F G : CategoryTheory.ComposableArrows C (HAdd.hAdd n 1)
                                                        ⊢ LE.le 0 (HAdd.hAdd n 1)
                                                      -/
                                                      /-
                                                        🎉 no goals
                                                      -/
def isoMkSucc {F G : ComposableArrows C (n + 1)} (α : F.obj' 0 ≅ G.obj' 0)
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
                           /-
                             C : Type u_1
                             inst✝ : CategoryTheory.Category.{?u.157328, u_1} C
                             n m : Nat
                             F✝ G✝ : CategoryTheory.ComposableArrows C n
                             F G : CategoryTheory.ComposableArrows C (HAdd.hAdd n 1)
                             α : CategoryTheory.Iso (F.obj' 0 ⋯) (G.obj' 0 ⋯)
                             β : CategoryTheory.Iso F.δ₀ G.δ₀
                             ⊢ LE.le 0 1
                           -/
                           /-
                             🎉 no goals
                           -/
                           /-
                             🎉 no goals
                           -/
                                        /-
                                          🎉 no goals
                                        -/
                                                               /-
                                                                 🎉 no goals
                                                               -/
    (β : F.δ₀ ≅ G.δ₀) (w : F.map' 0 1 ≫ app' β.hom 0 = α.hom ≫ G.map' 0 1) : F ≅ G where
                                                               /-
                                                                 🎉 no goals
                                                               -/
  hom := homMkSucc α.hom β.hom w
  inv := homMkSucc α.inv β.inv (by
    /-
      C : Type u_1
      inst✝ : CategoryTheory.Category.{?u.157328, u_1} C
      n m : Nat
      F✝ G✝ : CategoryTheory.ComposableArrows C n
      F G : CategoryTheory.ComposableArrows C (HAdd.hAdd n 1)
      α : CategoryTheory.Iso (F.obj' 0 ⋯) (G.obj' 0 ⋯)
      β : CategoryTheory.Iso F.δ₀ G.δ₀
      w : Eq (CategoryTheory.CategoryStruct.comp (F.map' 0 1 ⋯ ⋯) (CategoryTheory.Co …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map' 0 1 ⋯ ⋯) (CategoryTheory.Comp …
    -/
    rw [← cancel_epi α.hom, ← reassoc_of% w, α.hom_inv_id_assoc, β.hom_inv_id_app]
    /-
      C : Type u_1
      inst✝ : CategoryTheory.Category.{?u.157328, u_1} C
      n m : Nat
      F✝ G✝ : CategoryTheory.ComposableArrows C n
      F G : CategoryTheory.ComposableArrows C (HAdd.hAdd n 1)
      α : CategoryTheory.Iso (F.obj' 0 ⋯) (G.obj' 0 ⋯)
      β : CategoryTheory.Iso F.δ₀ G.δ₀
      w : Eq (CategoryTheory.CategoryStruct.comp (F.map' 0 1 ⋯ ⋯) (CategoryTheory.Co …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map' 0 1 ⋯ ⋯) (CategoryTheory.Cate …
    -/
    dsimp
    /-
      C : Type u_1
      inst✝ : CategoryTheory.Category.{?u.157328, u_1} C
      n m : Nat
      F✝ G✝ : CategoryTheory.ComposableArrows C n
      F G : CategoryTheory.ComposableArrows C (HAdd.hAdd n 1)
      α : CategoryTheory.Iso (F.obj' 0 ⋯) (G.obj' 0 ⋯)
      β : CategoryTheory.Iso F.δ₀ G.δ₀
      w : Eq (CategoryTheory.CategoryStruct.comp (F.map' 0 1 ⋯ ⋯) (CategoryTheory.Co …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.homOfLE ⋯)) (C …
    -/
    rw [comp_id])
    /-
      🎉 no goals
    -/
  hom_inv_id := by
    /-
      C : Type u_1
      inst✝ : CategoryTheory.Category.{?u.157328, u_1} C
      n m : Nat
      F✝ G✝ : CategoryTheory.ComposableArrows C n
      F G : CategoryTheory.ComposableArrows C (HAdd.hAdd n 1)
      α : CategoryTheory.Iso (F.obj' 0 ⋯) (G.obj' 0 ⋯)
      β : CategoryTheory.Iso F.δ₀ G.δ₀
      w : Eq (CategoryTheory.CategoryStruct.comp (F.map' 0 1 ⋯ ⋯) (CategoryTheory.Co …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.ComposableArrows.homM …
    -/
    apply hom_ext_succ
      /-
        case h₀
        C : Type u_1
        inst✝ : CategoryTheory.Category.{?u.157328, u_1} C
        n m : Nat
        F✝ G✝ : CategoryTheory.ComposableArrows C n
        F G : CategoryTheory.ComposableArrows C (HAdd.hAdd n 1)
        α : CategoryTheory.Iso (F.obj' 0 ⋯) (G.obj' 0 ⋯)
        β : CategoryTheory.Iso F.δ₀ G.δ₀
        w : Eq (CategoryTheory.CategoryStruct.comp (F.map' 0 1 ⋯ ⋯) (CategoryTheory.Co …
        ⊢ Eq (CategoryTheory.ComposableArrows.app' (CategoryTheory.CategoryStruct.comp …
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case h₁
        C : Type u_1
        inst✝ : CategoryTheory.Category.{?u.157328, u_1} C
        n m : Nat
        F✝ G✝ : CategoryTheory.ComposableArrows C n
        F G : CategoryTheory.ComposableArrows C (HAdd.hAdd n 1)
        α : CategoryTheory.Iso (F.obj' 0 ⋯) (G.obj' 0 ⋯)
        β : CategoryTheory.Iso F.δ₀ G.δ₀
        w : Eq (CategoryTheory.CategoryStruct.comp (F.map' 0 1 ⋯ ⋯) (CategoryTheory.Co …
        ⊢ Eq (CategoryTheory.ComposableArrows.δ₀Functor.map (CategoryTheory.CategorySt …
      -/
    · ext ⟨i, hi⟩
      /-
        case h₁.w.h.mk
        C : Type u_1
        inst✝ : CategoryTheory.Category.{?u.157328, u_1} C
        n m : Nat
        F✝ G✝ : CategoryTheory.ComposableArrows C n
        F G : CategoryTheory.ComposableArrows C (HAdd.hAdd n 1)
        α : CategoryTheory.Iso (F.obj' 0 ⋯) (G.obj' 0 ⋯)
        β : CategoryTheory.Iso F.δ₀ G.δ₀
        w : Eq (CategoryTheory.CategoryStruct.comp (F.map' 0 1 ⋯ ⋯) (CategoryTheory.Co …
        i : Nat
        hi : LT.lt i (HAdd.hAdd n 1)
        ⊢ Eq ((CategoryTheory.ComposableArrows.δ₀Functor.map (CategoryTheory.CategoryS …
      -/
      simp
      /-
        🎉 no goals
      -/
  inv_hom_id := by
    /-
      C : Type u_1
      inst✝ : CategoryTheory.Category.{?u.157328, u_1} C
      n m : Nat
      F✝ G✝ : CategoryTheory.ComposableArrows C n
      F G : CategoryTheory.ComposableArrows C (HAdd.hAdd n 1)
      α : CategoryTheory.Iso (F.obj' 0 ⋯) (G.obj' 0 ⋯)
      β : CategoryTheory.Iso F.δ₀ G.δ₀
      w : Eq (CategoryTheory.CategoryStruct.comp (F.map' 0 1 ⋯ ⋯) (CategoryTheory.Co …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.ComposableArrows.homM …
    -/
    apply hom_ext_succ
      /-
        case h₀
        C : Type u_1
        inst✝ : CategoryTheory.Category.{?u.157328, u_1} C
        n m : Nat
        F✝ G✝ : CategoryTheory.ComposableArrows C n
        F G : CategoryTheory.ComposableArrows C (HAdd.hAdd n 1)
        α : CategoryTheory.Iso (F.obj' 0 ⋯) (G.obj' 0 ⋯)
        β : CategoryTheory.Iso F.δ₀ G.δ₀
        w : Eq (CategoryTheory.CategoryStruct.comp (F.map' 0 1 ⋯ ⋯) (CategoryTheory.Co …
        ⊢ Eq (CategoryTheory.ComposableArrows.app' (CategoryTheory.CategoryStruct.comp …
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case h₁
        C : Type u_1
        inst✝ : CategoryTheory.Category.{?u.157328, u_1} C
        n m : Nat
        F✝ G✝ : CategoryTheory.ComposableArrows C n
        F G : CategoryTheory.ComposableArrows C (HAdd.hAdd n 1)
        α : CategoryTheory.Iso (F.obj' 0 ⋯) (G.obj' 0 ⋯)
        β : CategoryTheory.Iso F.δ₀ G.δ₀
        w : Eq (CategoryTheory.CategoryStruct.comp (F.map' 0 1 ⋯ ⋯) (CategoryTheory.Co …
        ⊢ Eq (CategoryTheory.ComposableArrows.δ₀Functor.map (CategoryTheory.CategorySt …
      -/
    · ext ⟨i, hi⟩
      /-
        case h₁.w.h.mk
        C : Type u_1
        inst✝ : CategoryTheory.Category.{?u.157328, u_1} C
        n m : Nat
        F✝ G✝ : CategoryTheory.ComposableArrows C n
        F G : CategoryTheory.ComposableArrows C (HAdd.hAdd n 1)
        α : CategoryTheory.Iso (F.obj' 0 ⋯) (G.obj' 0 ⋯)
        β : CategoryTheory.Iso F.δ₀ G.δ₀
        w : Eq (CategoryTheory.CategoryStruct.comp (F.map' 0 1 ⋯ ⋯) (CategoryTheory.Co …
        i : Nat
        hi : LT.lt i (HAdd.hAdd n 1)
        ⊢ Eq ((CategoryTheory.ComposableArrows.δ₀Functor.map (CategoryTheory.CategoryS …
      -/
      simp
      /-
        🎉 no goals
      -/


                                                        /-
                                                          C : Type u_1
                                                          inst✝ : CategoryTheory.Category.{?u.171197, u_1} C
                                                          n m : Nat
                                                          F✝ G✝ : CategoryTheory.ComposableArrows C n
                                                          F G : CategoryTheory.ComposableArrows C (HAdd.hAdd n 1)
                                                          ⊢ LE.le 0 (HAdd.hAdd n 1)
                                                        -/
                                                        /-
                                                          🎉 no goals
                                                        -/
lemma ext_succ {F G : ComposableArrows C (n + 1)} (h₀ : F.obj' 0 = G.obj' 0)
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
                           /-
                             C : Type u_1
                             inst✝ : CategoryTheory.Category.{?u.171197, u_1} C
                             n m : Nat
                             F✝ G✝ : CategoryTheory.ComposableArrows C n
                             F G : CategoryTheory.ComposableArrows C (HAdd.hAdd n 1)
                             h₀ : Eq (F.obj' 0 ⋯) (G.obj' 0 ⋯)
                             h : Eq F.δ₀ G.δ₀
                             ⊢ LE.le 0 1
                           -/
                           /-
                             🎉 no goals
                           -/
                           /-
                             🎉 no goals
                           -/
                                                     /-
                                                       🎉 no goals
                                                     -/
    (h : F.δ₀ = G.δ₀) (w : F.map' 0 1 = eqToHom h₀ ≫ G.map' 0 1 ≫
                                                     /-
                                                       🎉 no goals
                                                     -/
      eqToHom (Functor.congr_obj h.symm 0)) : F = G := by
  have : ∀ i, F.obj i = G.obj i := by
    intro ⟨i, hi⟩
    cases' i with i
    · exact h₀
    · exact Functor.congr_obj h ⟨i, by valid⟩
  exact Functor.ext_of_iso (isoMkSucc (eqToIso h₀) (eqToIso h) (by
      rw [w]
      dsimp [app']
      rw [eqToHom_app, assoc, assoc, eqToHom_trans, eqToHom_refl, comp_id])) this (by
    rintro ⟨i, hi⟩
    dsimp
    cases' i with i
    · erw [homMkSucc_app_zero]
    · rw [homMkSucc_app_succ]
      dsimp [app']
      rw [eqToHom_app])


lemma precomp_surjective (F : ComposableArrows C (n + 1)) :
    ∃ (F₀ : ComposableArrows C n) (X₀ : C) (f₀ : X₀ ⟶ F₀.left), F = F₀.precomp f₀ :=
            /-
              C : Type u_1
              inst✝ : CategoryTheory.Category.{u_2, u_1} C
              n : Nat
              F : CategoryTheory.ComposableArrows C (HAdd.hAdd n 1)
              ⊢ LE.le 0 1
            -/
            /-
              🎉 no goals
            -/
            /-
              🎉 no goals
            -/
                                         /-
                                           🎉 no goals
                                         -/
  ⟨F.δ₀, _, F.map' 0 1, ext_succ rfl (by simp) (by simp)⟩
                                                   /-
                                                     🎉 no goals
                                                   -/


y.ComposableArrows C 2
              ⊢ LE.le 0 2
            -/
            /-
              🎉 no goals
            -/
                       /-
                         🎉 no goals
                       -/
                                         /-
                                           🎉 no goals
                                         -/
                                                    /-
                                                      🎉 no goals
                                                    -/
                                                                      /-
                                                                        🎉 no goals
                                                                      -/
    (app₀ : f.obj' 0 ⟶ g.obj' 0) (app₁ : f.obj' 1 ⟶ g.obj' 1) (app₂ : f.obj' 2 ⟶ g.obj' 2)
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/
          /-
            C : Type u_1
            inst✝ : CategoryTheory.Category.{?u.188809, u_1} C
            n m : Nat
            F G : CategoryTheory.ComposableArrows C n
            f g : CategoryTheory.ComposableArrows C 2
            app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
            app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
            app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
            ⊢ LE.le 0 1
          -/
          /-
            🎉 no goals
          -/
          /-
            🎉 no goals
          -/
                                     /-
                                       🎉 no goals
                                     -/
    (w₀ : f.map' 0 1 ≫ app₁ = app₀ ≫ g.map' 0 1)
                                     /-
                                       🎉 no goals
                                     -/
          /-
            C : Type u_1
            inst✝ : CategoryTheory.Category.{?u.188809, u_1} C
            n m : Nat
            F G : CategoryTheory.ComposableArrows C n
            f g : CategoryTheory.ComposableArrows C 2
            app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
            app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
            app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
            w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁) (CategoryTh …
            ⊢ LE.le 1 2
          -/
          /-
            🎉 no goals
          -/
          /-
            🎉 no goals
          -/
                                     /-
                                       🎉 no goals
                                     -/
    (w₁ : f.map' 1 2 ≫ app₂ = app₁ ≫ g.map' 1 2)
                                     /-
                                       🎉 no goals
                                     -/

/-- Constructor for morphisms in `ComposableArrows C 2`. -/
def homMk₂ : f ⟶ g := homMkSucc app₀ (homMk₁ app₁ app₂ w₁) w₀


        /-
                         🎉 no goals
                       -/
                                         /-
                                           🎉 no goals
                                         -/
                                                    /-
                                                      🎉 no goals
                                                    -/
                                                                      /-
                                                                        🎉 no goals
                                                                      -/
    (app₀ : f.obj' 0 ⟶ g.obj' 0) (app₁ : f.obj' 1 ⟶ g.obj' 1) (app₂ : f.obj' 2 ⟶ g.obj' 2)
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/
          /-
            C : Type u_1
            inst✝ : CategoryTheory.Category.{?u.190905, u_1} C
            n m : Nat
            F G : CategoryTheory.ComposableArrows C n
            f g : CategoryTheory.ComposableArrows C 2
            app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
            app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
            app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
            ⊢ LE.le 0 1
          -/
          /-
            🎉 no goals
          -/
          /-
            🎉 no goals
          -/
                                     /-
                                       🎉 no goals
                                     -/
    (w₀ : f.map' 0 1 ≫ app₁ = app₀ ≫ g.map' 0 1)
                                     /-
                                       🎉 no goals
                                     -/
          /-
            C : Type u_1
            inst✝ : CategoryTheory.Category.{?u.190905, u_1} C
            n m : Nat
            F G : CategoryTheory.ComposableArrows C n
            f g : CategoryTheory.ComposableArrows C 2
            app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
            app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
            app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
            w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁) (CategoryTh …
            ⊢ LE.le 1 2
          -/
          /-
            🎉 no goals
          -/
          /-
            🎉 no goals
          -/
                                     /-
                                       🎉 no goals
                                     -/
    (w₁ : f.map' 1 2 ≫ app₂ = app₁ ≫ g.map' 1 2)
                                     /-
                                       🎉 no goals
                                     -/

/-- Constructor for morphisms in `ComposableArrows C 2`. -/
def homMk₂ : f ⟶ g := homMkSucc app₀ (homMk₁ app₁ app₂ w₁) w₀

@[simp]
lemma homMk₂_app_zero : (homMk₂ app₀ app₁ app₂ w₀ w₁).app 0 = app₀ := rfl


                  /-
                                           🎉 no goals
                                         -/
                                                    /-
                                                      🎉 no goals
                                                    -/
                                                                      /-
                                                                        🎉 no goals
                                                                      -/
    (app₀ : f.obj' 0 ⟶ g.obj' 0) (app₁ : f.obj' 1 ⟶ g.obj' 1) (app₂ : f.obj' 2 ⟶ g.obj' 2)
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/
          /-
            C : Type u_1
            inst✝ : CategoryTheory.Category.{?u.192832, u_1} C
            n m : Nat
            F G : CategoryTheory.ComposableArrows C n
            f g : CategoryTheory.ComposableArrows C 2
            app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
            app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
            app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
            ⊢ LE.le 0 1
          -/
          /-
            🎉 no goals
          -/
          /-
            🎉 no goals
          -/
                                     /-
                                       🎉 no goals
                                     -/
    (w₀ : f.map' 0 1 ≫ app₁ = app₀ ≫ g.map' 0 1)
                                     /-
                                       🎉 no goals
                                     -/
          /-
            C : Type u_1
            inst✝ : CategoryTheory.Category.{?u.192832, u_1} C
            n m : Nat
            F G : CategoryTheory.ComposableArrows C n
            f g : CategoryTheory.ComposableArrows C 2
            app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
            app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
            app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
            w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁) (CategoryTh …
            ⊢ LE.le 1 2
          -/
          /-
            🎉 no goals
          -/
          /-
            🎉 no goals
          -/
                                     /-
                                       🎉 no goals
                                     -/
    (w₁ : f.map' 1 2 ≫ app₂ = app₁ ≫ g.map' 1 2)
                                     /-
                                       🎉 no goals
                                     -/

/-- Constructor for morphisms in `ComposableArrows C 2`. -/
def homMk₂ : f ⟶ g := homMkSucc app₀ (homMk₁ app₁ app₂ w₁) w₀

@[simp]
lemma homMk₂_app_zero : (homMk₂ app₀ app₁ app₂ w₀ w₁).app 0 = app₀ := rfl

@[simp]
lemma homMk₂_app_one : (homMk₂ app₀ app₁ app₂ w₀ w₁).app 1 = app₁ := rfl


                     -/
                                                    /-
                                                      🎉 no goals
                                                    -/
                                                                      /-
                                                                        🎉 no goals
                                                                      -/
    (app₀ : f.obj' 0 ⟶ g.obj' 0) (app₁ : f.obj' 1 ⟶ g.obj' 1) (app₂ : f.obj' 2 ⟶ g.obj' 2)
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/
          /-
            C : Type u_1
            inst✝ : CategoryTheory.Category.{?u.194771, u_1} C
            n m : Nat
            F G : CategoryTheory.ComposableArrows C n
            f g : CategoryTheory.ComposableArrows C 2
            app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
            app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
            app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
            ⊢ LE.le 0 1
          -/
          /-
            🎉 no goals
          -/
          /-
            🎉 no goals
          -/
                                     /-
                                       🎉 no goals
                                     -/
    (w₀ : f.map' 0 1 ≫ app₁ = app₀ ≫ g.map' 0 1)
                                     /-
                                       🎉 no goals
                                     -/
          /-
            C : Type u_1
            inst✝ : CategoryTheory.Category.{?u.194771, u_1} C
            n m : Nat
            F G : CategoryTheory.ComposableArrows C n
            f g : CategoryTheory.ComposableArrows C 2
            app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
            app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
            app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
            w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁) (CategoryTh …
            ⊢ LE.le 1 2
          -/
          /-
            🎉 no goals
          -/
          /-
            🎉 no goals
          -/
                                     /-
                                       🎉 no goals
                                     -/
    (w₁ : f.map' 1 2 ≫ app₂ = app₁ ≫ g.map' 1 2)
                                     /-
                                       🎉 no goals
                                     -/

/-- Constructor for morphisms in `ComposableArrows C 2`. -/
def homMk₂ : f ⟶ g := homMkSucc app₀ (homMk₁ app₁ app₂ w₁) w₀

@[simp]
lemma homMk₂_app_zero : (homMk₂ app₀ app₁ app₂ w₀ w₁).app 0 = app₀ := rfl

@[simp]
lemma homMk₂_app_one : (homMk₂ app₀ app₁ app₂ w₀ w₁).app 1 = app₁ := rfl

@[simp]
                                                                /-
                                                                  C : Type u_1
                                                                  inst✝ : CategoryTheory.Category.{?u.194771, u_1} C
                                                                  n m : Nat
                                                                  F G : CategoryTheory.ComposableArrows C n
                                                                  f g : CategoryTheory.ComposableArrows C 2
                                                                  app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
                                                                  app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
                                                                  app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
                                                                  w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁) (CategoryTh …
                                                                  w₁ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 1 2 ⋯ ⋯) app₂) (CategoryTh …
                                                                  ⊢ LT.lt 2 (HAdd.hAdd 2 1)
                                                                -/
lemma homMk₂_app_two : (homMk₂ app₀ app₁ app₂ w₀ w₁).app ⟨2, by valid⟩ = app₂ := rfl
                                                                /-
                                                                  🎉 no goals
                                                                -/


@[ext]
lemma hom_ext₂ {f g : ComposableArrows C 2} {φ φ' : f ⟶ g}
          /-
            C : Type u_1
            inst✝ : CategoryTheory.Category.{?u.196831, u_1} C
            n m : Nat
            F G : CategoryTheory.ComposableArrows C n
            f g : CategoryTheory.ComposableArrows C 2
            φ φ' : Quiver.Hom f g
            ⊢ LE.le 0 2
          -/
          /-
            🎉 no goals
          -/
                     /-
                       🎉 no goals
                     -/
                                      /-
                                        🎉 no goals
                                      -/
                                                 /-
                                                   🎉 no goals
                                                 -/
                                                                  /-
                                                                    🎉 no goals
                                                                  -/
    (h₀ : app' φ 0 = app' φ' 0) (h₁ : app' φ 1 = app' φ' 1) (h₂ : app' φ 2 = app' φ' 2) :
                                                                             /-
                                                                               🎉 no goals
                                                                             -/
    φ = φ' :=
  hom_ext_succ h₀ (hom_ext₁ h₁ h₂)


/-- Constructor for isomorphisms in `ComposableArrows C 2`. -/
@[simps]
def isoMk₂ {f g : ComposableArrows C 2}
            /-
              C : Type u_1
              inst✝ : CategoryTheory.Category.{?u.198223, u_1} C
              n m : Nat
              F G : CategoryTheory.ComposableArrows C n
              f g : CategoryTheory.ComposableArrows C 2
              ⊢ LE.le 0 2
            -/
            /-
              🎉 no goals
            -/
                       /-
                         🎉 no goals
                       -/
                                         /-
                                           🎉 no goals
                                         -/
                                                    /-
                                                      🎉 no goals
                                                    -/
                                                                      /-
                                                                        🎉 no goals
                                                                      -/
    (app₀ : f.obj' 0 ≅ g.obj' 0) (app₁ : f.obj' 1 ≅ g.obj' 1) (app₂ : f.obj' 2 ≅ g.obj' 2)
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/
          /-
            C : Type u_1
            inst✝ : CategoryTheory.Category.{?u.198223, u_1} C
            n m : Nat
            F G : CategoryTheory.ComposableArrows C n
            f g : CategoryTheory.ComposableArrows C 2
            app₀ : CategoryTheory.Iso (f.obj' 0 ⋯) (g.obj' 0 ⋯)
            app₁ : CategoryTheory.Iso (f.obj' 1 ⋯) (g.obj' 1 ⋯)
            app₂ : CategoryTheory.Iso (f.obj' 2 ⋯) (g.obj' 2 ⋯)
            ⊢ LE.le 0 1
          -/
          /-
            🎉 no goals
          -/
          /-
            🎉 no goals
          -/
                                             /-
                                               🎉 no goals
                                             -/
    (w₀ : f.map' 0 1 ≫ app₁.hom = app₀.hom ≫ g.map' 0 1)
                                             /-
                                               🎉 no goals
                                             -/
          /-
            C : Type u_1
            inst✝ : CategoryTheory.Category.{?u.198223, u_1} C
            n m : Nat
            F G : CategoryTheory.ComposableArrows C n
            f g : CategoryTheory.ComposableArrows C 2
            app₀ : CategoryTheory.Iso (f.obj' 0 ⋯) (g.obj' 0 ⋯)
            app₁ : CategoryTheory.Iso (f.obj' 1 ⋯) (g.obj' 1 ⋯)
            app₂ : CategoryTheory.Iso (f.obj' 2 ⋯) (g.obj' 2 ⋯)
            w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁.hom) (Catego …
            ⊢ LE.le 1 2
          -/
          /-
            🎉 no goals
          -/
          /-
            🎉 no goals
          -/
                                             /-
                                               🎉 no goals
                                             -/
    (w₁ : f.map' 1 2 ≫ app₂.hom = app₁.hom ≫ g.map' 1 2) : f ≅ g where
                                             /-
                                               🎉 no goals
                                             -/
  hom := homMk₂ app₀.hom app₁.hom app₂.hom w₀ w₁
  inv := homMk₂ app₀.inv app₁.inv app₂.inv
    (by rw [← cancel_epi app₀.hom, ← reassoc_of% w₀, app₁.hom_inv_id,
      comp_id, app₀.hom_inv_id_assoc])
    (by rw [← cancel_epi app₁.hom, ← reassoc_of% w₁, app₂.hom_inv_id,
      comp_id, app₁.hom_inv_id_assoc])


lemma ext₂ {f g : ComposableArrows C 2}
          /-
            C : Type u_1
            inst✝ : CategoryTheory.Category.{?u.212146, u_1} C
            n m : Nat
            F G : CategoryTheory.ComposableArrows C n
            f g : CategoryTheory.ComposableArrows C 2
            ⊢ LE.le 0 2
          -/
          /-
            🎉 no goals
          -/
                     /-
                       🎉 no goals
                     -/
                                     /-
                                       🎉 no goals
                                     -/
                                                /-
                                                  🎉 no goals
                                                -/
                                                                /-
                                                                  🎉 no goals
                                                                -/
    (h₀ : f.obj' 0 = g.obj' 0) (h₁ : f.obj' 1 = g.obj' 1) (h₂ : f.obj' 2 = g.obj' 2)
                                                                           /-
                                                                             🎉 no goals
                                                                           -/
          /-
            C : Type u_1
            inst✝ : CategoryTheory.Category.{?u.212146, u_1} C
            n m : Nat
            F G : CategoryTheory.ComposableArrows C n
            f g : CategoryTheory.ComposableArrows C 2
            h₀ : Eq (f.obj' 0 ⋯) (g.obj' 0 ⋯)
            h₁ : Eq (f.obj' 1 ⋯) (g.obj' 1 ⋯)
            h₂ : Eq (f.obj' 2 ⋯) (g.obj' 2 ⋯)
            ⊢ LE.le 0 1
          -/
          /-
            🎉 no goals
          -/
          /-
            🎉 no goals
          -/
                                    /-
                                      🎉 no goals
                                    -/
    (w₀ : f.map' 0 1 = eqToHom h₀ ≫ g.map' 0 1 ≫ eqToHom h₁.symm)
                                    /-
                                      🎉 no goals
                                    -/
          /-
            C : Type u_1
            inst✝ : CategoryTheory.Category.{?u.212146, u_1} C
            n m : Nat
            F G : CategoryTheory.ComposableArrows C n
            f g : CategoryTheory.ComposableArrows C 2
            h₀ : Eq (f.obj' 0 ⋯) (g.obj' 0 ⋯)
            h₁ : Eq (f.obj' 1 ⋯) (g.obj' 1 ⋯)
            h₂ : Eq (f.obj' 2 ⋯) (g.obj' 2 ⋯)
            w₀ : Eq (f.map' 0 1 ⋯ ⋯) (CategoryTheory.CategoryStruct.comp (CategoryTheory.e …
            ⊢ LE.le 1 2
          -/
          /-
            🎉 no goals
          -/
          /-
            🎉 no goals
          -/
                                    /-
                                      🎉 no goals
                                    -/
    (w₁ : f.map' 1 2 = eqToHom h₁ ≫ g.map' 1 2 ≫ eqToHom h₂.symm) : f = g :=
                                    /-
                                      🎉 no goals
                                    -/
  ext_succ h₀ (ext₁ h₁ h₂ w₁) w₀


lemma mk₂_surjective (X : ComposableArrows C 2) :
    ∃ (X₀ X₁ X₂ : C) (f₀ : X₀ ⟶ X₁) (f₁ : X₁ ⟶ X₂), X = mk₂ f₀ f₁ :=
            /-
              C : Type u_1
              inst✝ : CategoryTheory.Category.{u_2, u_1} C
              X : CategoryTheory.ComposableArrows C 2
              ⊢ LE.le 0 1
            -/
            /-
              🎉 no goals
            -/
            /-
              🎉 no goals
            -/
                        /-
                          🎉 no goals
                        -/
                        /-
                          🎉 no goals
                        -/
                                                         /-
                                                           🎉 no goals
                                                         -/
  ⟨_, _, _, X.map' 0 1, X.map' 1 2, ext₂ rfl rfl rfl (by simp) (by simp)⟩
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


ls
          -/
                     /-
                       🎉 no goals
                     -/
                                       /-
                                         🎉 no goals
                                       -/
                                                  /-
                                                    🎉 no goals
                                                  -/
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
  (app₀ : f.obj' 0 ⟶ g.obj' 0) (app₁ : f.obj' 1 ⟶ g.obj' 1) (app₂ : f.obj' 2 ⟶ g.obj' 2)
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/
          /-
            C : Type u_1
            inst✝ : CategoryTheory.Category.{?u.224460, u_1} C
            n m : Nat
            F G : CategoryTheory.ComposableArrows C n
            f g : CategoryTheory.ComposableArrows C 3
            app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
            app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
            app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
            ⊢ LE.le 3 3
          -/
          /-
            🎉 no goals
          -/
  (app₃ : f.obj' 3 ⟶ g.obj' 3)
                     /-
                       🎉 no goals
                     -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.224460, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 3
          app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          ⊢ LE.le 0 1
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
                                   /-
                                     🎉 no goals
                                   -/
  (w₀ : f.map' 0 1 ≫ app₁ = app₀ ≫ g.map' 0 1)
                                   /-
                                     🎉 no goals
                                   -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.224460, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 3
          app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁) (CategoryTh …
          ⊢ LE.le 1 2
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
                                   /-
                                     🎉 no goals
                                   -/
  (w₁ : f.map' 1 2 ≫ app₂ = app₁ ≫ g.map' 1 2)
                                   /-
                                     🎉 no goals
                                   -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.224460, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 3
          app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁) (CategoryTh …
          w₁ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 1 2 ⋯ ⋯) app₂) (CategoryTh …
          ⊢ LE.le 2 3
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
                                   /-
                                     🎉 no goals
                                   -/
  (w₂ : f.map' 2 3 ≫ app₃ = app₂ ≫ g.map' 2 3)
                                   /-
                                     🎉 no goals
                                   -/

/-- Constructor for morphisms in `ComposableArrows C 3`. -/
def homMk₃ : f ⟶ g := homMkSucc app₀ (homMk₂ app₁ app₂ app₃ w₁ w₂) w₀


                                 🎉 no goals
                                       -/
                                                  /-
                                                    🎉 no goals
                                                  -/
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
  (app₀ : f.obj' 0 ⟶ g.obj' 0) (app₁ : f.obj' 1 ⟶ g.obj' 1) (app₂ : f.obj' 2 ⟶ g.obj' 2)
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/
          /-
            C : Type u_1
            inst✝ : CategoryTheory.Category.{?u.227453, u_1} C
            n m : Nat
            F G : CategoryTheory.ComposableArrows C n
            f g : CategoryTheory.ComposableArrows C 3
            app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
            app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
            app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
            ⊢ LE.le 3 3
          -/
          /-
            🎉 no goals
          -/
  (app₃ : f.obj' 3 ⟶ g.obj' 3)
                     /-
                       🎉 no goals
                     -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.227453, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 3
          app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          ⊢ LE.le 0 1
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
                                   /-
                                     🎉 no goals
                                   -/
  (w₀ : f.map' 0 1 ≫ app₁ = app₀ ≫ g.map' 0 1)
                                   /-
                                     🎉 no goals
                                   -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.227453, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 3
          app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁) (CategoryTh …
          ⊢ LE.le 1 2
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
                                   /-
                                     🎉 no goals
                                   -/
  (w₁ : f.map' 1 2 ≫ app₂ = app₁ ≫ g.map' 1 2)
                                   /-
                                     🎉 no goals
                                   -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.227453, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 3
          app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁) (CategoryTh …
          w₁ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 1 2 ⋯ ⋯) app₂) (CategoryTh …
          ⊢ LE.le 2 3
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
                                   /-
                                     🎉 no goals
                                   -/
  (w₂ : f.map' 2 3 ≫ app₃ = app₂ ≫ g.map' 2 3)
                                   /-
                                     🎉 no goals
                                   -/

/-- Constructor for morphisms in `ComposableArrows C 3`. -/
def homMk₃ : f ⟶ g := homMkSucc app₀ (homMk₂ app₁ app₂ app₃ w₁ w₂) w₀

@[simp]
lemma homMk₃_app_zero : (homMk₃ app₀ app₁ app₂ app₃ w₀ w₁ w₂).app 0 = app₀ := rfl


                            /-
                                                    🎉 no goals
                                                  -/
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
  (app₀ : f.obj' 0 ⟶ g.obj' 0) (app₁ : f.obj' 1 ⟶ g.obj' 1) (app₂ : f.obj' 2 ⟶ g.obj' 2)
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/
          /-
            C : Type u_1
            inst✝ : CategoryTheory.Category.{?u.230249, u_1} C
            n m : Nat
            F G : CategoryTheory.ComposableArrows C n
            f g : CategoryTheory.ComposableArrows C 3
            app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
            app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
            app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
            ⊢ LE.le 3 3
          -/
          /-
            🎉 no goals
          -/
  (app₃ : f.obj' 3 ⟶ g.obj' 3)
                     /-
                       🎉 no goals
                     -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.230249, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 3
          app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          ⊢ LE.le 0 1
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
                                   /-
                                     🎉 no goals
                                   -/
  (w₀ : f.map' 0 1 ≫ app₁ = app₀ ≫ g.map' 0 1)
                                   /-
                                     🎉 no goals
                                   -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.230249, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 3
          app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁) (CategoryTh …
          ⊢ LE.le 1 2
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
                                   /-
                                     🎉 no goals
                                   -/
  (w₁ : f.map' 1 2 ≫ app₂ = app₁ ≫ g.map' 1 2)
                                   /-
                                     🎉 no goals
                                   -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.230249, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 3
          app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁) (CategoryTh …
          w₁ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 1 2 ⋯ ⋯) app₂) (CategoryTh …
          ⊢ LE.le 2 3
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
                                   /-
                                     🎉 no goals
                                   -/
  (w₂ : f.map' 2 3 ≫ app₃ = app₂ ≫ g.map' 2 3)
                                   /-
                                     🎉 no goals
                                   -/

/-- Constructor for morphisms in `ComposableArrows C 3`. -/
def homMk₃ : f ⟶ g := homMkSucc app₀ (homMk₂ app₁ app₂ app₃ w₁ w₂) w₀

@[simp]
lemma homMk₃_app_zero : (homMk₃ app₀ app₁ app₂ app₃ w₀ w₁ w₂).app 0 = app₀ := rfl

@[simp]
lemma homMk₃_app_one : (homMk₃ app₀ app₁ app₂ app₃ w₀ w₁ w₂).app 1 = app₁ := rfl


                                     -/
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
  (app₀ : f.obj' 0 ⟶ g.obj' 0) (app₁ : f.obj' 1 ⟶ g.obj' 1) (app₂ : f.obj' 2 ⟶ g.obj' 2)
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/
          /-
            C : Type u_1
            inst✝ : CategoryTheory.Category.{?u.233055, u_1} C
            n m : Nat
            F G : CategoryTheory.ComposableArrows C n
            f g : CategoryTheory.ComposableArrows C 3
            app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
            app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
            app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
            ⊢ LE.le 3 3
          -/
          /-
            🎉 no goals
          -/
  (app₃ : f.obj' 3 ⟶ g.obj' 3)
                     /-
                       🎉 no goals
                     -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.233055, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 3
          app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          ⊢ LE.le 0 1
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
                                   /-
                                     🎉 no goals
                                   -/
  (w₀ : f.map' 0 1 ≫ app₁ = app₀ ≫ g.map' 0 1)
                                   /-
                                     🎉 no goals
                                   -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.233055, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 3
          app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁) (CategoryTh …
          ⊢ LE.le 1 2
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
                                   /-
                                     🎉 no goals
                                   -/
  (w₁ : f.map' 1 2 ≫ app₂ = app₁ ≫ g.map' 1 2)
                                   /-
                                     🎉 no goals
                                   -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.233055, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 3
          app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁) (CategoryTh …
          w₁ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 1 2 ⋯ ⋯) app₂) (CategoryTh …
          ⊢ LE.le 2 3
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
                                   /-
                                     🎉 no goals
                                   -/
  (w₂ : f.map' 2 3 ≫ app₃ = app₂ ≫ g.map' 2 3)
                                   /-
                                     🎉 no goals
                                   -/

/-- Constructor for morphisms in `ComposableArrows C 3`. -/
def homMk₃ : f ⟶ g := homMkSucc app₀ (homMk₂ app₁ app₂ app₃ w₁ w₂) w₀

@[simp]
lemma homMk₃_app_zero : (homMk₃ app₀ app₁ app₂ app₃ w₀ w₁ w₂).app 0 = app₀ := rfl

@[simp]
lemma homMk₃_app_one : (homMk₃ app₀ app₁ app₂ app₃ w₀ w₁ w₂).app 1 = app₁ := rfl

@[simp]
                                                                        /-
                                                                          C : Type u_1
                                                                          inst✝ : CategoryTheory.Category.{?u.233055, u_1} C
                                                                          n m : Nat
                                                                          F G : CategoryTheory.ComposableArrows C n
                                                                          f g : CategoryTheory.ComposableArrows C 3
                                                                          app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
                                                                          app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
                                                                          app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
                                                                          app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
                                                                          w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁) (CategoryTh …
                                                                          w₁ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 1 2 ⋯ ⋯) app₂) (CategoryTh …
                                                                          w₂ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 2 3 ⋯ ⋯) app₃) (CategoryTh …
                                                                          ⊢ LT.lt 2 (HAdd.hAdd 3 1)
                                                                        -/
lemma homMk₃_app_two : (homMk₃ app₀ app₁ app₂ app₃ w₀ w₁ w₂).app ⟨2, by valid⟩ = app₂ :=
                                                                        /-
                                                                          🎉 no goals
                                                                        -/
  rfl


                                                     🎉 no goals
                                                                    -/
  (app₀ : f.obj' 0 ⟶ g.obj' 0) (app₁ : f.obj' 1 ⟶ g.obj' 1) (app₂ : f.obj' 2 ⟶ g.obj' 2)
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/
          /-
            C : Type u_1
            inst✝ : CategoryTheory.Category.{?u.235994, u_1} C
            n m : Nat
            F G : CategoryTheory.ComposableArrows C n
            f g : CategoryTheory.ComposableArrows C 3
            app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
            app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
            app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
            ⊢ LE.le 3 3
          -/
          /-
            🎉 no goals
          -/
  (app₃ : f.obj' 3 ⟶ g.obj' 3)
                     /-
                       🎉 no goals
                     -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.235994, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 3
          app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          ⊢ LE.le 0 1
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
                                   /-
                                     🎉 no goals
                                   -/
  (w₀ : f.map' 0 1 ≫ app₁ = app₀ ≫ g.map' 0 1)
                                   /-
                                     🎉 no goals
                                   -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.235994, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 3
          app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁) (CategoryTh …
          ⊢ LE.le 1 2
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
                                   /-
                                     🎉 no goals
                                   -/
  (w₁ : f.map' 1 2 ≫ app₂ = app₁ ≫ g.map' 1 2)
                                   /-
                                     🎉 no goals
                                   -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.235994, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 3
          app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁) (CategoryTh …
          w₁ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 1 2 ⋯ ⋯) app₂) (CategoryTh …
          ⊢ LE.le 2 3
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
                                   /-
                                     🎉 no goals
                                   -/
  (w₂ : f.map' 2 3 ≫ app₃ = app₂ ≫ g.map' 2 3)
                                   /-
                                     🎉 no goals
                                   -/

/-- Constructor for morphisms in `ComposableArrows C 3`. -/
def homMk₃ : f ⟶ g := homMkSucc app₀ (homMk₂ app₁ app₂ app₃ w₁ w₂) w₀

@[simp]
lemma homMk₃_app_zero : (homMk₃ app₀ app₁ app₂ app₃ w₀ w₁ w₂).app 0 = app₀ := rfl

@[simp]
lemma homMk₃_app_one : (homMk₃ app₀ app₁ app₂ app₃ w₀ w₁ w₂).app 1 = app₁ := rfl

@[simp]
lemma homMk₃_app_two : (homMk₃ app₀ app₁ app₂ app₃ w₀ w₁ w₂).app ⟨2, by valid⟩ = app₂ :=
  rfl

@[simp]
                                                                          /-
                                                                            C : Type u_1
                                                                            inst✝ : CategoryTheory.Category.{?u.235994, u_1} C
                                                                            n m : Nat
                                                                            F G : CategoryTheory.ComposableArrows C n
                                                                            f g : CategoryTheory.ComposableArrows C 3
                                                                            app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
                                                                            app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
                                                                            app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
                                                                            app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
                                                                            w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁) (CategoryTh …
                                                                            w₁ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 1 2 ⋯ ⋯) app₂) (CategoryTh …
                                                                            w₂ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 2 3 ⋯ ⋯) app₃) (CategoryTh …
                                                                            ⊢ LT.lt 3 (HAdd.hAdd 3 1)
                                                                          -/
lemma homMk₃_app_three : (homMk₃ app₀ app₁ app₂ app₃ w₀ w₁ w₂).app ⟨3, by valid⟩ = app₃ :=
                                                                          /-
                                                                            🎉 no goals
                                                                          -/
  rfl


@[ext]
lemma hom_ext₃ {f g : ComposableArrows C 3} {φ φ' : f ⟶ g}
          /-
            C : Type u_1
            inst✝ : CategoryTheory.Category.{?u.238933, u_1} C
            n m : Nat
            F G : CategoryTheory.ComposableArrows C n
            f g : CategoryTheory.ComposableArrows C 3
            φ φ' : Quiver.Hom f g
            ⊢ LE.le 0 3
          -/
          /-
            🎉 no goals
          -/
                     /-
                       🎉 no goals
                     -/
                                      /-
                                        🎉 no goals
                                      -/
                                                 /-
                                                   🎉 no goals
                                                 -/
                                                                  /-
                                                                    🎉 no goals
                                                                  -/
    (h₀ : app' φ 0 = app' φ' 0) (h₁ : app' φ 1 = app' φ' 1) (h₂ : app' φ 2 = app' φ' 2)
                                                                             /-
                                                                               🎉 no goals
                                                                             -/
          /-
            C : Type u_1
            inst✝ : CategoryTheory.Category.{?u.238933, u_1} C
            n m : Nat
            F G : CategoryTheory.ComposableArrows C n
            f g : CategoryTheory.ComposableArrows C 3
            φ φ' : Quiver.Hom f g
            h₀ : Eq (CategoryTheory.ComposableArrows.app' φ 0 ⋯) (CategoryTheory.Composabl …
            h₁ : Eq (CategoryTheory.ComposableArrows.app' φ 1 ⋯) (CategoryTheory.Composabl …
            h₂ : Eq (CategoryTheory.ComposableArrows.app' φ 2 ⋯) (CategoryTheory.Composabl …
            ⊢ LE.le 3 3
          -/
          /-
            🎉 no goals
          -/
    (h₃ : app' φ 3 = app' φ' 3) :
                     /-
                       🎉 no goals
                     -/
    φ = φ' :=
  hom_ext_succ h₀ (hom_ext₂ h₁ h₂ h₃)


/-- Constructor for isomorphisms in `ComposableArrows C 3`. -/
@[simps]
def isoMk₃ {f g : ComposableArrows C 3}
            /-
              C : Type u_1
              inst✝ : CategoryTheory.Category.{?u.240676, u_1} C
              n m : Nat
              F G : CategoryTheory.ComposableArrows C n
              f g : CategoryTheory.ComposableArrows C 3
              ⊢ LE.le 0 3
            -/
            /-
              🎉 no goals
            -/
                       /-
                         🎉 no goals
                       -/
                                         /-
                                           🎉 no goals
                                         -/
                                                    /-
                                                      🎉 no goals
                                                    -/
                                                                      /-
                                                                        🎉 no goals
                                                                      -/
    (app₀ : f.obj' 0 ≅ g.obj' 0) (app₁ : f.obj' 1 ≅ g.obj' 1) (app₂ : f.obj' 2 ≅ g.obj' 2)
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/
            /-
              C : Type u_1
              inst✝ : CategoryTheory.Category.{?u.240676, u_1} C
              n m : Nat
              F G : CategoryTheory.ComposableArrows C n
              f g : CategoryTheory.ComposableArrows C 3
              app₀ : CategoryTheory.Iso (f.obj' 0 ⋯) (g.obj' 0 ⋯)
              app₁ : CategoryTheory.Iso (f.obj' 1 ⋯) (g.obj' 1 ⋯)
              app₂ : CategoryTheory.Iso (f.obj' 2 ⋯) (g.obj' 2 ⋯)
              ⊢ LE.le 3 3
            -/
            /-
              🎉 no goals
            -/
    (app₃ : f.obj' 3 ≅ g.obj' 3)
                       /-
                         🎉 no goals
                       -/
          /-
            C : Type u_1
            inst✝ : CategoryTheory.Category.{?u.240676, u_1} C
            n m : Nat
            F G : CategoryTheory.ComposableArrows C n
            f g : CategoryTheory.ComposableArrows C 3
            app₀ : CategoryTheory.Iso (f.obj' 0 ⋯) (g.obj' 0 ⋯)
            app₁ : CategoryTheory.Iso (f.obj' 1 ⋯) (g.obj' 1 ⋯)
            app₂ : CategoryTheory.Iso (f.obj' 2 ⋯) (g.obj' 2 ⋯)
            app₃ : CategoryTheory.Iso (f.obj' 3 ⋯) (g.obj' 3 ⋯)
            ⊢ LE.le 0 1
          -/
          /-
            🎉 no goals
          -/
          /-
            🎉 no goals
          -/
                                             /-
                                               🎉 no goals
                                             -/
    (w₀ : f.map' 0 1 ≫ app₁.hom = app₀.hom ≫ g.map' 0 1)
                                             /-
                                               🎉 no goals
                                             -/
          /-
            C : Type u_1
            inst✝ : CategoryTheory.Category.{?u.240676, u_1} C
            n m : Nat
            F G : CategoryTheory.ComposableArrows C n
            f g : CategoryTheory.ComposableArrows C 3
            app₀ : CategoryTheory.Iso (f.obj' 0 ⋯) (g.obj' 0 ⋯)
            app₁ : CategoryTheory.Iso (f.obj' 1 ⋯) (g.obj' 1 ⋯)
            app₂ : CategoryTheory.Iso (f.obj' 2 ⋯) (g.obj' 2 ⋯)
            app₃ : CategoryTheory.Iso (f.obj' 3 ⋯) (g.obj' 3 ⋯)
            w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁.hom) (Catego …
            ⊢ LE.le 1 2
          -/
          /-
            🎉 no goals
          -/
          /-
            🎉 no goals
          -/
                                             /-
                                               🎉 no goals
                                             -/
    (w₁ : f.map' 1 2 ≫ app₂.hom = app₁.hom ≫ g.map' 1 2)
                                             /-
                                               🎉 no goals
                                             -/
          /-
            C : Type u_1
            inst✝ : CategoryTheory.Category.{?u.240676, u_1} C
            n m : Nat
            F G : CategoryTheory.ComposableArrows C n
            f g : CategoryTheory.ComposableArrows C 3
            app₀ : CategoryTheory.Iso (f.obj' 0 ⋯) (g.obj' 0 ⋯)
            app₁ : CategoryTheory.Iso (f.obj' 1 ⋯) (g.obj' 1 ⋯)
            app₂ : CategoryTheory.Iso (f.obj' 2 ⋯) (g.obj' 2 ⋯)
            app₃ : CategoryTheory.Iso (f.obj' 3 ⋯) (g.obj' 3 ⋯)
            w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁.hom) (Catego …
            w₁ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 1 2 ⋯ ⋯) app₂.hom) (Catego …
            ⊢ LE.le 2 3
          -/
          /-
            🎉 no goals
          -/
          /-
            🎉 no goals
          -/
                                             /-
                                               🎉 no goals
                                             -/
    (w₂ : f.map' 2 3 ≫ app₃.hom = app₂.hom ≫ g.map' 2 3) : f ≅ g where
                                             /-
                                               🎉 no goals
                                             -/
  hom := homMk₃ app₀.hom app₁.hom app₂.hom app₃.hom w₀ w₁ w₂
  inv := homMk₃ app₀.inv app₁.inv app₂.inv app₃.inv
    (by rw [← cancel_epi app₀.hom, ← reassoc_of% w₀, app₁.hom_inv_id,
      comp_id, app₀.hom_inv_id_assoc])
    (by rw [← cancel_epi app₁.hom, ← reassoc_of% w₁, app₂.hom_inv_id,
      comp_id, app₁.hom_inv_id_assoc])
    (by rw [← cancel_epi app₂.hom, ← reassoc_of% w₂, app₃.hom_inv_id,
      comp_id, app₂.hom_inv_id_assoc])


lemma ext₃ {f g : ComposableArrows C 3}
          /-
            C : Type u_1
            inst✝ : CategoryTheory.Category.{?u.261370, u_1} C
            n m : Nat
            F G : CategoryTheory.ComposableArrows C n
            f g : CategoryTheory.ComposableArrows C 3
            ⊢ LE.le 0 3
          -/
          /-
            🎉 no goals
          -/
                     /-
                       🎉 no goals
                     -/
                                     /-
                                       🎉 no goals
                                     -/
                                                /-
                                                  🎉 no goals
                                                -/
                                                                /-
                                                                  🎉 no goals
                                                                -/
    (h₀ : f.obj' 0 = g.obj' 0) (h₁ : f.obj' 1 = g.obj' 1) (h₂ : f.obj' 2 = g.obj' 2)
                                                                           /-
                                                                             🎉 no goals
                                                                           -/
          /-
            C : Type u_1
            inst✝ : CategoryTheory.Category.{?u.261370, u_1} C
            n m : Nat
            F G : CategoryTheory.ComposableArrows C n
            f g : CategoryTheory.ComposableArrows C 3
            h₀ : Eq (f.obj' 0 ⋯) (g.obj' 0 ⋯)
            h₁ : Eq (f.obj' 1 ⋯) (g.obj' 1 ⋯)
            h₂ : Eq (f.obj' 2 ⋯) (g.obj' 2 ⋯)
            ⊢ LE.le 3 3
          -/
          /-
            🎉 no goals
          -/
    (h₃ : f.obj' 3 = g.obj' 3)
                     /-
                       🎉 no goals
                     -/
          /-
            C : Type u_1
            inst✝ : CategoryTheory.Category.{?u.261370, u_1} C
            n m : Nat
            F G : CategoryTheory.ComposableArrows C n
            f g : CategoryTheory.ComposableArrows C 3
            h₀ : Eq (f.obj' 0 ⋯) (g.obj' 0 ⋯)
            h₁ : Eq (f.obj' 1 ⋯) (g.obj' 1 ⋯)
            h₂ : Eq (f.obj' 2 ⋯) (g.obj' 2 ⋯)
            h₃ : Eq (f.obj' 3 ⋯) (g.obj' 3 ⋯)
            ⊢ LE.le 0 1
          -/
          /-
            🎉 no goals
          -/
          /-
            🎉 no goals
          -/
                                    /-
                                      🎉 no goals
                                    -/
    (w₀ : f.map' 0 1 = eqToHom h₀ ≫ g.map' 0 1 ≫ eqToHom h₁.symm)
                                    /-
                                      🎉 no goals
                                    -/
          /-
            C : Type u_1
            inst✝ : CategoryTheory.Category.{?u.261370, u_1} C
            n m : Nat
            F G : CategoryTheory.ComposableArrows C n
            f g : CategoryTheory.ComposableArrows C 3
            h₀ : Eq (f.obj' 0 ⋯) (g.obj' 0 ⋯)
            h₁ : Eq (f.obj' 1 ⋯) (g.obj' 1 ⋯)
            h₂ : Eq (f.obj' 2 ⋯) (g.obj' 2 ⋯)
            h₃ : Eq (f.obj' 3 ⋯) (g.obj' 3 ⋯)
            w₀ : Eq (f.map' 0 1 ⋯ ⋯) (CategoryTheory.CategoryStruct.comp (CategoryTheory.e …
            ⊢ LE.le 1 2
          -/
          /-
            🎉 no goals
          -/
          /-
            🎉 no goals
          -/
                                    /-
                                      🎉 no goals
                                    -/
    (w₁ : f.map' 1 2 = eqToHom h₁ ≫ g.map' 1 2 ≫ eqToHom h₂.symm)
                                    /-
                                      🎉 no goals
                                    -/
          /-
            C : Type u_1
            inst✝ : CategoryTheory.Category.{?u.261370, u_1} C
            n m : Nat
            F G : CategoryTheory.ComposableArrows C n
            f g : CategoryTheory.ComposableArrows C 3
            h₀ : Eq (f.obj' 0 ⋯) (g.obj' 0 ⋯)
            h₁ : Eq (f.obj' 1 ⋯) (g.obj' 1 ⋯)
            h₂ : Eq (f.obj' 2 ⋯) (g.obj' 2 ⋯)
            h₃ : Eq (f.obj' 3 ⋯) (g.obj' 3 ⋯)
            w₀ : Eq (f.map' 0 1 ⋯ ⋯) (CategoryTheory.CategoryStruct.comp (CategoryTheory.e …
            w₁ : Eq (f.map' 1 2 ⋯ ⋯) (CategoryTheory.CategoryStruct.comp (CategoryTheory.e …
            ⊢ LE.le 2 3
          -/
          /-
            🎉 no goals
          -/
          /-
            🎉 no goals
          -/
                                    /-
                                      🎉 no goals
                                    -/
    (w₂ : f.map' 2 3 = eqToHom h₂ ≫ g.map' 2 3 ≫ eqToHom h₃.symm) : f = g :=
                                    /-
                                      🎉 no goals
                                    -/
  ext_succ h₀ (ext₂ h₁ h₂ h₃ w₁ w₂) w₀


lemma mk₃_surjective (X : ComposableArrows C 3) :
    ∃ (X₀ X₁ X₂ X₃ : C) (f₀ : X₀ ⟶ X₁) (f₁ : X₁ ⟶ X₂) (f₂ : X₂ ⟶ X₃), X = mk₃ f₀ f₁ f₂ :=
               /-
                 C : Type u_1
                 inst✝ : CategoryTheory.Category.{u_2, u_1} C
                 X : CategoryTheory.ComposableArrows C 3
                 ⊢ LE.le 0 1
               -/
               /-
                 🎉 no goals
               -/
               /-
                 🎉 no goals
               -/
                           /-
                             🎉 no goals
                           -/
                           /-
                             🎉 no goals
                           -/
                                       /-
                                         🎉 no goals
                                       -/
  ⟨_, _, _, _, X.map' 0 1, X.map' 1 2, X.map' 2 3,
                                       /-
                                         🎉 no goals
                                       -/
                             /-
                               C : Type u_1
                               inst✝ : CategoryTheory.Category.{u_2, u_1} C
                               X : CategoryTheory.ComposableArrows C 3
                               ⊢ Eq (X.map' 0 1 ⋯ ⋯) (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqTo …
                             -/
                             /-
                               🎉 no goals
                             -/
                                       /-
                                         🎉 no goals
                                       -/
    ext₃ rfl rfl rfl rfl (by simp) (by simp) (by simp)⟩
                                                 /-
                                                   🎉 no goals
                                                 -/


        -/
                                       /-
                                         🎉 no goals
                                       -/
                                                  /-
                                                    🎉 no goals
                                                  -/
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
  (app₀ : f.obj' 0 ⟶ g.obj' 0) (app₁ : f.obj' 1 ⟶ g.obj' 1) (app₂ : f.obj' 2 ⟶ g.obj' 2)
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/
          /-
            C : Type u_1
            inst✝ : CategoryTheory.Category.{?u.284252, u_1} C
            n m : Nat
            F G : CategoryTheory.ComposableArrows C n
            f g : CategoryTheory.ComposableArrows C 4
            app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
            app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
            app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
            ⊢ LE.le 3 4
          -/
          /-
            🎉 no goals
          -/
                     /-
                       🎉 no goals
                     -/
                                       /-
                                         🎉 no goals
                                       -/
  (app₃ : f.obj' 3 ⟶ g.obj' 3) (app₄ : f.obj' 4 ⟶ g.obj' 4)
                                                  /-
                                                    🎉 no goals
                                                  -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.284252, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 4
          app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          app₄ : Quiver.Hom (f.obj' 4 ⋯) (g.obj' 4 ⋯)
          ⊢ LE.le 0 1
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
                                   /-
                                     🎉 no goals
                                   -/
  (w₀ : f.map' 0 1 ≫ app₁ = app₀ ≫ g.map' 0 1)
                                   /-
                                     🎉 no goals
                                   -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.284252, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 4
          app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          app₄ : Quiver.Hom (f.obj' 4 ⋯) (g.obj' 4 ⋯)
          w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁) (CategoryTh …
          ⊢ LE.le 1 2
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
                                   /-
                                     🎉 no goals
                                   -/
  (w₁ : f.map' 1 2 ≫ app₂ = app₁ ≫ g.map' 1 2)
                                   /-
                                     🎉 no goals
                                   -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.284252, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 4
          app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          app₄ : Quiver.Hom (f.obj' 4 ⋯) (g.obj' 4 ⋯)
          w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁) (CategoryTh …
          w₁ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 1 2 ⋯ ⋯) app₂) (CategoryTh …
          ⊢ LE.le 2 3
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
                                   /-
                                     🎉 no goals
                                   -/
  (w₂ : f.map' 2 3 ≫ app₃ = app₂ ≫ g.map' 2 3)
                                   /-
                                     🎉 no goals
                                   -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.284252, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 4
          app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          app₄ : Quiver.Hom (f.obj' 4 ⋯) (g.obj' 4 ⋯)
          w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁) (CategoryTh …
          w₁ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 1 2 ⋯ ⋯) app₂) (CategoryTh …
          w₂ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 2 3 ⋯ ⋯) app₃) (CategoryTh …
          ⊢ LE.le 3 4
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
                                   /-
                                     🎉 no goals
                                   -/
  (w₃ : f.map' 3 4 ≫ app₄ = app₃ ≫ g.map' 3 4)
                                   /-
                                     🎉 no goals
                                   -/

/-- Constructor for morphisms in `ComposableArrows C 4`. -/
def homMk₄ : f ⟶ g := homMkSucc app₀ (homMk₃ app₁ app₂ app₃ app₄ w₁ w₂ w₃) w₀


                                     /-
                                                    🎉 no goals
                                                  -/
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
  (app₀ : f.obj' 0 ⟶ g.obj' 0) (app₁ : f.obj' 1 ⟶ g.obj' 1) (app₂ : f.obj' 2 ⟶ g.obj' 2)
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/
          /-
            C : Type u_1
            inst✝ : CategoryTheory.Category.{?u.288138, u_1} C
            n m : Nat
            F G : CategoryTheory.ComposableArrows C n
            f g : CategoryTheory.ComposableArrows C 4
            app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
            app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
            app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
            ⊢ LE.le 3 4
          -/
          /-
            🎉 no goals
          -/
                     /-
                       🎉 no goals
                     -/
                                       /-
                                         🎉 no goals
                                       -/
  (app₃ : f.obj' 3 ⟶ g.obj' 3) (app₄ : f.obj' 4 ⟶ g.obj' 4)
                                                  /-
                                                    🎉 no goals
                                                  -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.288138, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 4
          app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          app₄ : Quiver.Hom (f.obj' 4 ⋯) (g.obj' 4 ⋯)
          ⊢ LE.le 0 1
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
                                   /-
                                     🎉 no goals
                                   -/
  (w₀ : f.map' 0 1 ≫ app₁ = app₀ ≫ g.map' 0 1)
                                   /-
                                     🎉 no goals
                                   -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.288138, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 4
          app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          app₄ : Quiver.Hom (f.obj' 4 ⋯) (g.obj' 4 ⋯)
          w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁) (CategoryTh …
          ⊢ LE.le 1 2
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
                                   /-
                                     🎉 no goals
                                   -/
  (w₁ : f.map' 1 2 ≫ app₂ = app₁ ≫ g.map' 1 2)
                                   /-
                                     🎉 no goals
                                   -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.288138, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 4
          app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          app₄ : Quiver.Hom (f.obj' 4 ⋯) (g.obj' 4 ⋯)
          w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁) (CategoryTh …
          w₁ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 1 2 ⋯ ⋯) app₂) (CategoryTh …
          ⊢ LE.le 2 3
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
                                   /-
                                     🎉 no goals
                                   -/
  (w₂ : f.map' 2 3 ≫ app₃ = app₂ ≫ g.map' 2 3)
                                   /-
                                     🎉 no goals
                                   -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.288138, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 4
          app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          app₄ : Quiver.Hom (f.obj' 4 ⋯) (g.obj' 4 ⋯)
          w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁) (CategoryTh …
          w₁ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 1 2 ⋯ ⋯) app₂) (CategoryTh …
          w₂ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 2 3 ⋯ ⋯) app₃) (CategoryTh …
          ⊢ LE.le 3 4
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
                                   /-
                                     🎉 no goals
                                   -/
  (w₃ : f.map' 3 4 ≫ app₄ = app₃ ≫ g.map' 3 4)
                                   /-
                                     🎉 no goals
                                   -/

/-- Constructor for morphisms in `ComposableArrows C 4`. -/
def homMk₄ : f ⟶ g := homMkSucc app₀ (homMk₃ app₁ app₂ app₃ app₄ w₁ w₂ w₃) w₀

@[simp]
lemma homMk₄_app_zero : (homMk₄ app₀ app₁ app₂ app₃ app₄ w₀ w₁ w₂ w₃).app 0 = app₀ := rfl


                                 -/
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
  (app₀ : f.obj' 0 ⟶ g.obj' 0) (app₁ : f.obj' 1 ⟶ g.obj' 1) (app₂ : f.obj' 2 ⟶ g.obj' 2)
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/
          /-
            C : Type u_1
            inst✝ : CategoryTheory.Category.{?u.291803, u_1} C
            n m : Nat
            F G : CategoryTheory.ComposableArrows C n
            f g : CategoryTheory.ComposableArrows C 4
            app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
            app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
            app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
            ⊢ LE.le 3 4
          -/
          /-
            🎉 no goals
          -/
                     /-
                       🎉 no goals
                     -/
                                       /-
                                         🎉 no goals
                                       -/
  (app₃ : f.obj' 3 ⟶ g.obj' 3) (app₄ : f.obj' 4 ⟶ g.obj' 4)
                                                  /-
                                                    🎉 no goals
                                                  -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.291803, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 4
          app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          app₄ : Quiver.Hom (f.obj' 4 ⋯) (g.obj' 4 ⋯)
          ⊢ LE.le 0 1
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
                                   /-
                                     🎉 no goals
                                   -/
  (w₀ : f.map' 0 1 ≫ app₁ = app₀ ≫ g.map' 0 1)
                                   /-
                                     🎉 no goals
                                   -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.291803, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 4
          app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          app₄ : Quiver.Hom (f.obj' 4 ⋯) (g.obj' 4 ⋯)
          w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁) (CategoryTh …
          ⊢ LE.le 1 2
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
                                   /-
                                     🎉 no goals
                                   -/
  (w₁ : f.map' 1 2 ≫ app₂ = app₁ ≫ g.map' 1 2)
                                   /-
                                     🎉 no goals
                                   -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.291803, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 4
          app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          app₄ : Quiver.Hom (f.obj' 4 ⋯) (g.obj' 4 ⋯)
          w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁) (CategoryTh …
          w₁ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 1 2 ⋯ ⋯) app₂) (CategoryTh …
          ⊢ LE.le 2 3
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
                                   /-
                                     🎉 no goals
                                   -/
  (w₂ : f.map' 2 3 ≫ app₃ = app₂ ≫ g.map' 2 3)
                                   /-
                                     🎉 no goals
                                   -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.291803, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 4
          app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          app₄ : Quiver.Hom (f.obj' 4 ⋯) (g.obj' 4 ⋯)
          w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁) (CategoryTh …
          w₁ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 1 2 ⋯ ⋯) app₂) (CategoryTh …
          w₂ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 2 3 ⋯ ⋯) app₃) (CategoryTh …
          ⊢ LE.le 3 4
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
                                   /-
                                     🎉 no goals
                                   -/
  (w₃ : f.map' 3 4 ≫ app₄ = app₃ ≫ g.map' 3 4)
                                   /-
                                     🎉 no goals
                                   -/

/-- Constructor for morphisms in `ComposableArrows C 4`. -/
def homMk₄ : f ⟶ g := homMkSucc app₀ (homMk₃ app₁ app₂ app₃ app₄ w₁ w₂ w₃) w₀

@[simp]
lemma homMk₄_app_zero : (homMk₄ app₀ app₁ app₂ app₃ app₄ w₀ w₁ w₂ w₃).app 0 = app₀ := rfl

@[simp]
lemma homMk₄_app_one : (homMk₄ app₀ app₁ app₂ app₃ app₄ w₀ w₁ w₂ w₃).app 1 = app₁ := rfl


                                                       🎉 no goals
                                                                    -/
  (app₀ : f.obj' 0 ⟶ g.obj' 0) (app₁ : f.obj' 1 ⟶ g.obj' 1) (app₂ : f.obj' 2 ⟶ g.obj' 2)
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/
          /-
            C : Type u_1
            inst✝ : CategoryTheory.Category.{?u.295478, u_1} C
            n m : Nat
            F G : CategoryTheory.ComposableArrows C n
            f g : CategoryTheory.ComposableArrows C 4
            app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
            app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
            app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
            ⊢ LE.le 3 4
          -/
          /-
            🎉 no goals
          -/
                     /-
                       🎉 no goals
                     -/
                                       /-
                                         🎉 no goals
                                       -/
  (app₃ : f.obj' 3 ⟶ g.obj' 3) (app₄ : f.obj' 4 ⟶ g.obj' 4)
                                                  /-
                                                    🎉 no goals
                                                  -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.295478, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 4
          app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          app₄ : Quiver.Hom (f.obj' 4 ⋯) (g.obj' 4 ⋯)
          ⊢ LE.le 0 1
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
                                   /-
                                     🎉 no goals
                                   -/
  (w₀ : f.map' 0 1 ≫ app₁ = app₀ ≫ g.map' 0 1)
                                   /-
                                     🎉 no goals
                                   -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.295478, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 4
          app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          app₄ : Quiver.Hom (f.obj' 4 ⋯) (g.obj' 4 ⋯)
          w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁) (CategoryTh …
          ⊢ LE.le 1 2
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
                                   /-
                                     🎉 no goals
                                   -/
  (w₁ : f.map' 1 2 ≫ app₂ = app₁ ≫ g.map' 1 2)
                                   /-
                                     🎉 no goals
                                   -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.295478, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 4
          app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          app₄ : Quiver.Hom (f.obj' 4 ⋯) (g.obj' 4 ⋯)
          w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁) (CategoryTh …
          w₁ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 1 2 ⋯ ⋯) app₂) (CategoryTh …
          ⊢ LE.le 2 3
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
                                   /-
                                     🎉 no goals
                                   -/
  (w₂ : f.map' 2 3 ≫ app₃ = app₂ ≫ g.map' 2 3)
                                   /-
                                     🎉 no goals
                                   -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.295478, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 4
          app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          app₄ : Quiver.Hom (f.obj' 4 ⋯) (g.obj' 4 ⋯)
          w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁) (CategoryTh …
          w₁ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 1 2 ⋯ ⋯) app₂) (CategoryTh …
          w₂ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 2 3 ⋯ ⋯) app₃) (CategoryTh …
          ⊢ LE.le 3 4
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
                                   /-
                                     🎉 no goals
                                   -/
  (w₃ : f.map' 3 4 ≫ app₄ = app₃ ≫ g.map' 3 4)
                                   /-
                                     🎉 no goals
                                   -/

/-- Constructor for morphisms in `ComposableArrows C 4`. -/
def homMk₄ : f ⟶ g := homMkSucc app₀ (homMk₃ app₁ app₂ app₃ app₄ w₁ w₂ w₃) w₀

@[simp]
lemma homMk₄_app_zero : (homMk₄ app₀ app₁ app₂ app₃ app₄ w₀ w₁ w₂ w₃).app 0 = app₀ := rfl

@[simp]
lemma homMk₄_app_one : (homMk₄ app₀ app₁ app₂ app₃ app₄ w₀ w₁ w₂ w₃).app 1 = app₁ := rfl

@[simp]
lemma homMk₄_app_two :
                                                             /-
                                                               C : Type u_1
                                                               inst✝ : CategoryTheory.Category.{?u.295478, u_1} C
                                                               n m : Nat
                                                               F G : CategoryTheory.ComposableArrows C n
                                                               f g : CategoryTheory.ComposableArrows C 4
                                                               app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
                                                               app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
                                                               app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
                                                               app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
                                                               app₄ : Quiver.Hom (f.obj' 4 ⋯) (g.obj' 4 ⋯)
                                                               w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁) (CategoryTh …
                                                               w₁ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 1 2 ⋯ ⋯) app₂) (CategoryTh …
                                                               w₂ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 2 3 ⋯ ⋯) app₃) (CategoryTh …
                                                               w₃ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 3 4 ⋯ ⋯) app₄) (CategoryTh …
                                                               ⊢ LT.lt 2 (HAdd.hAdd 4 1)
                                                             -/
    (homMk₄ app₀ app₁ app₂ app₃ app₄ w₀ w₁ w₂ w₃).app ⟨2, by valid⟩ = app₂ := rfl
                                                             /-
                                                               🎉 no goals
                                                             -/


(app₀ : f.obj' 0 ⟶ g.obj' 0) (app₁ : f.obj' 1 ⟶ g.obj' 1) (app₂ : f.obj' 2 ⟶ g.obj' 2)
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/
          /-
            C : Type u_1
            inst✝ : CategoryTheory.Category.{?u.299284, u_1} C
            n m : Nat
            F G : CategoryTheory.ComposableArrows C n
            f g : CategoryTheory.ComposableArrows C 4
            app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
            app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
            app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
            ⊢ LE.le 3 4
          -/
          /-
            🎉 no goals
          -/
                     /-
                       🎉 no goals
                     -/
                                       /-
                                         🎉 no goals
                                       -/
  (app₃ : f.obj' 3 ⟶ g.obj' 3) (app₄ : f.obj' 4 ⟶ g.obj' 4)
                                                  /-
                                                    🎉 no goals
                                                  -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.299284, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 4
          app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          app₄ : Quiver.Hom (f.obj' 4 ⋯) (g.obj' 4 ⋯)
          ⊢ LE.le 0 1
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
                                   /-
                                     🎉 no goals
                                   -/
  (w₀ : f.map' 0 1 ≫ app₁ = app₀ ≫ g.map' 0 1)
                                   /-
                                     🎉 no goals
                                   -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.299284, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 4
          app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          app₄ : Quiver.Hom (f.obj' 4 ⋯) (g.obj' 4 ⋯)
          w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁) (CategoryTh …
          ⊢ LE.le 1 2
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
                                   /-
                                     🎉 no goals
                                   -/
  (w₁ : f.map' 1 2 ≫ app₂ = app₁ ≫ g.map' 1 2)
                                   /-
                                     🎉 no goals
                                   -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.299284, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 4
          app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          app₄ : Quiver.Hom (f.obj' 4 ⋯) (g.obj' 4 ⋯)
          w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁) (CategoryTh …
          w₁ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 1 2 ⋯ ⋯) app₂) (CategoryTh …
          ⊢ LE.le 2 3
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
                                   /-
                                     🎉 no goals
                                   -/
  (w₂ : f.map' 2 3 ≫ app₃ = app₂ ≫ g.map' 2 3)
                                   /-
                                     🎉 no goals
                                   -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.299284, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 4
          app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          app₄ : Quiver.Hom (f.obj' 4 ⋯) (g.obj' 4 ⋯)
          w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁) (CategoryTh …
          w₁ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 1 2 ⋯ ⋯) app₂) (CategoryTh …
          w₂ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 2 3 ⋯ ⋯) app₃) (CategoryTh …
          ⊢ LE.le 3 4
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
                                   /-
                                     🎉 no goals
                                   -/
  (w₃ : f.map' 3 4 ≫ app₄ = app₃ ≫ g.map' 3 4)
                                   /-
                                     🎉 no goals
                                   -/

/-- Constructor for morphisms in `ComposableArrows C 4`. -/
def homMk₄ : f ⟶ g := homMkSucc app₀ (homMk₃ app₁ app₂ app₃ app₄ w₁ w₂ w₃) w₀

@[simp]
lemma homMk₄_app_zero : (homMk₄ app₀ app₁ app₂ app₃ app₄ w₀ w₁ w₂ w₃).app 0 = app₀ := rfl

@[simp]
lemma homMk₄_app_one : (homMk₄ app₀ app₁ app₂ app₃ app₄ w₀ w₁ w₂ w₃).app 1 = app₁ := rfl

@[simp]
lemma homMk₄_app_two :
    (homMk₄ app₀ app₁ app₂ app₃ app₄ w₀ w₁ w₂ w₃).app ⟨2, by valid⟩ = app₂ := rfl

@[simp]
lemma homMk₄_app_three :
                                                             /-
                                                               C : Type u_1
                                                               inst✝ : CategoryTheory.Category.{?u.299284, u_1} C
                                                               n m : Nat
                                                               F G : CategoryTheory.ComposableArrows C n
                                                               f g : CategoryTheory.ComposableArrows C 4
                                                               app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
                                                               app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
                                                               app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
                                                               app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
                                                               app₄ : Quiver.Hom (f.obj' 4 ⋯) (g.obj' 4 ⋯)
                                                               w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁) (CategoryTh …
                                                               w₁ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 1 2 ⋯ ⋯) app₂) (CategoryTh …
                                                               w₂ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 2 3 ⋯ ⋯) app₃) (CategoryTh …
                                                               w₃ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 3 4 ⋯ ⋯) app₄) (CategoryTh …
                                                               ⊢ LT.lt 3 (HAdd.hAdd 4 1)
                                                             -/
    (homMk₄ app₀ app₁ app₂ app₃ app₄ w₀ w₁ w₂ w₃).app ⟨3, by valid⟩ = app₃ := rfl
                                                             /-
                                                               🎉 no goals
                                                             -/


                                  /-
                                                                                 🎉 no goals
                                                                               -/
          /-
            C : Type u_1
            inst✝ : CategoryTheory.Category.{?u.303102, u_1} C
            n m : Nat
            F G : CategoryTheory.ComposableArrows C n
            f g : CategoryTheory.ComposableArrows C 4
            app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
            app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
            app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
            ⊢ LE.le 3 4
          -/
          /-
            🎉 no goals
          -/
                     /-
                       🎉 no goals
                     -/
                                       /-
                                         🎉 no goals
                                       -/
  (app₃ : f.obj' 3 ⟶ g.obj' 3) (app₄ : f.obj' 4 ⟶ g.obj' 4)
                                                  /-
                                                    🎉 no goals
                                                  -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.303102, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 4
          app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          app₄ : Quiver.Hom (f.obj' 4 ⋯) (g.obj' 4 ⋯)
          ⊢ LE.le 0 1
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
                                   /-
                                     🎉 no goals
                                   -/
  (w₀ : f.map' 0 1 ≫ app₁ = app₀ ≫ g.map' 0 1)
                                   /-
                                     🎉 no goals
                                   -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.303102, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 4
          app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          app₄ : Quiver.Hom (f.obj' 4 ⋯) (g.obj' 4 ⋯)
          w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁) (CategoryTh …
          ⊢ LE.le 1 2
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
                                   /-
                                     🎉 no goals
                                   -/
  (w₁ : f.map' 1 2 ≫ app₂ = app₁ ≫ g.map' 1 2)
                                   /-
                                     🎉 no goals
                                   -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.303102, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 4
          app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          app₄ : Quiver.Hom (f.obj' 4 ⋯) (g.obj' 4 ⋯)
          w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁) (CategoryTh …
          w₁ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 1 2 ⋯ ⋯) app₂) (CategoryTh …
          ⊢ LE.le 2 3
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
                                   /-
                                     🎉 no goals
                                   -/
  (w₂ : f.map' 2 3 ≫ app₃ = app₂ ≫ g.map' 2 3)
                                   /-
                                     🎉 no goals
                                   -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.303102, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 4
          app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          app₄ : Quiver.Hom (f.obj' 4 ⋯) (g.obj' 4 ⋯)
          w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁) (CategoryTh …
          w₁ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 1 2 ⋯ ⋯) app₂) (CategoryTh …
          w₂ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 2 3 ⋯ ⋯) app₃) (CategoryTh …
          ⊢ LE.le 3 4
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
                                   /-
                                     🎉 no goals
                                   -/
  (w₃ : f.map' 3 4 ≫ app₄ = app₃ ≫ g.map' 3 4)
                                   /-
                                     🎉 no goals
                                   -/

/-- Constructor for morphisms in `ComposableArrows C 4`. -/
def homMk₄ : f ⟶ g := homMkSucc app₀ (homMk₃ app₁ app₂ app₃ app₄ w₁ w₂ w₃) w₀

@[simp]
lemma homMk₄_app_zero : (homMk₄ app₀ app₁ app₂ app₃ app₄ w₀ w₁ w₂ w₃).app 0 = app₀ := rfl

@[simp]
lemma homMk₄_app_one : (homMk₄ app₀ app₁ app₂ app₃ app₄ w₀ w₁ w₂ w₃).app 1 = app₁ := rfl

@[simp]
lemma homMk₄_app_two :
    (homMk₄ app₀ app₁ app₂ app₃ app₄ w₀ w₁ w₂ w₃).app ⟨2, by valid⟩ = app₂ := rfl

@[simp]
lemma homMk₄_app_three :
    (homMk₄ app₀ app₁ app₂ app₃ app₄ w₀ w₁ w₂ w₃).app ⟨3, by valid⟩ = app₃ := rfl

@[simp]
lemma homMk₄_app_four :
                                                             /-
                                                               C : Type u_1
                                                               inst✝ : CategoryTheory.Category.{?u.303102, u_1} C
                                                               n m : Nat
                                                               F G : CategoryTheory.ComposableArrows C n
                                                               f g : CategoryTheory.ComposableArrows C 4
                                                               app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
                                                               app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
                                                               app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
                                                               app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
                                                               app₄ : Quiver.Hom (f.obj' 4 ⋯) (g.obj' 4 ⋯)
                                                               w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁) (CategoryTh …
                                                               w₁ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 1 2 ⋯ ⋯) app₂) (CategoryTh …
                                                               w₂ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 2 3 ⋯ ⋯) app₃) (CategoryTh …
                                                               w₃ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 3 4 ⋯ ⋯) app₄) (CategoryTh …
                                                               ⊢ LT.lt 4 (HAdd.hAdd 4 1)
                                                             -/
    (homMk₄ app₀ app₁ app₂ app₃ app₄ w₀ w₁ w₂ w₃).app ⟨4, by valid⟩ = app₄ := rfl
                                                             /-
                                                               🎉 no goals
                                                             -/


@[ext]
lemma hom_ext₄ {f g : ComposableArrows C 4} {φ φ' : f ⟶ g}
          /-
            C : Type u_1
            inst✝ : CategoryTheory.Category.{?u.306920, u_1} C
            n m : Nat
            F G : CategoryTheory.ComposableArrows C n
            f g : CategoryTheory.ComposableArrows C 4
            φ φ' : Quiver.Hom f g
            ⊢ LE.le 0 4
          -/
          /-
            🎉 no goals
          -/
                     /-
                       🎉 no goals
                     -/
                                      /-
                                        🎉 no goals
                                      -/
                                                 /-
                                                   🎉 no goals
                                                 -/
                                                                  /-
                                                                    🎉 no goals
                                                                  -/
    (h₀ : app' φ 0 = app' φ' 0) (h₁ : app' φ 1 = app' φ' 1) (h₂ : app' φ 2 = app' φ' 2)
                                                                             /-
                                                                               🎉 no goals
                                                                             -/
          /-
            C : Type u_1
            inst✝ : CategoryTheory.Category.{?u.306920, u_1} C
            n m : Nat
            F G : CategoryTheory.ComposableArrows C n
            f g : CategoryTheory.ComposableArrows C 4
            φ φ' : Quiver.Hom f g
            h₀ : Eq (CategoryTheory.ComposableArrows.app' φ 0 ⋯) (CategoryTheory.Composabl …
            h₁ : Eq (CategoryTheory.ComposableArrows.app' φ 1 ⋯) (CategoryTheory.Composabl …
            h₂ : Eq (CategoryTheory.ComposableArrows.app' φ 2 ⋯) (CategoryTheory.Composabl …
            ⊢ LE.le 3 4
          -/
          /-
            🎉 no goals
          -/
                     /-
                       🎉 no goals
                     -/
                                      /-
                                        🎉 no goals
                                      -/
    (h₃ : app' φ 3 = app' φ' 3) (h₄ : app' φ 4 = app' φ' 4) :
                                                 /-
                                                   🎉 no goals
                                                 -/
    φ = φ' :=
  hom_ext_succ h₀ (hom_ext₃ h₁ h₂ h₃ h₄)


lemma map'_inv_eq_inv_map' {n m : ℕ} (h : n+1 ≤ m) {f g : ComposableArrows C m}
           /-
             C : Type u_1
             inst✝ : CategoryTheory.Category.{?u.309014, u_1} C
             n✝ m✝ : Nat
             F G : CategoryTheory.ComposableArrows C n✝
             n m : Nat
             h : LE.le (HAdd.hAdd n 1) m
             f g : CategoryTheory.ComposableArrows C m
             ⊢ LE.le n m
           -/
           /-
             🎉 no goals
           -/
                      /-
                        🎉 no goals
                      -/
                                        /-
                                          🎉 no goals
                                        -/
    (app : f.obj' n ≅ g.obj' n) (app' : f.obj' (n+1) ≅ g.obj' (n+1))
                                                       /-
                                                         🎉 no goals
                                                       -/
         /-
           C : Type u_1
           inst✝ : CategoryTheory.Category.{?u.309014, u_1} C
           n✝ m✝ : Nat
           F G : CategoryTheory.ComposableArrows C n✝
           n m : Nat
           h : LE.le (HAdd.hAdd n 1) m
           f g : CategoryTheory.ComposableArrows C m
           app : CategoryTheory.Iso (f.obj' n ⋯) (g.obj' n ⋯)
           app' : CategoryTheory.Iso (f.obj' (HAdd.hAdd n 1) h) (g.obj' (HAdd.hAdd n 1) h)
           ⊢ LE.le n (HAdd.hAdd n 1)
         -/
         /-
           🎉 no goals
         -/
         /-
           🎉 no goals
         -/
                                               /-
                                                 🎉 no goals
                                               -/
    (w : f.map' n (n+1) ≫ app'.hom = app.hom ≫ g.map' n (n+1)) :
                                               /-
                                                 🎉 no goals
                                               -/
    /-
      C : Type u_1
      inst✝ : CategoryTheory.Category.{?u.309014, u_1} C
      n✝ m✝ : Nat
      F G : CategoryTheory.ComposableArrows C n✝
      n m : Nat
      h : LE.le (HAdd.hAdd n 1) m
      f g : CategoryTheory.ComposableArrows C m
      app : CategoryTheory.Iso (f.obj' n ⋯) (g.obj' n ⋯)
      app' : CategoryTheory.Iso (f.obj' (HAdd.hAdd n 1) h) (g.obj' (HAdd.hAdd n 1) h)
      w : Eq (CategoryTheory.CategoryStruct.comp (f.map' n (HAdd.hAdd n 1) ⋯ h) app' …
      ⊢ LE.le n (HAdd.hAdd n 1)
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
                                          /-
                                            🎉 no goals
                                          -/
    map' g n (n+1) ≫ app'.inv = app.inv ≫ map' f n (n+1) := by
                                          /-
                                            🎉 no goals
                                          -/
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    n m : Nat
    h : LE.le (HAdd.hAdd n 1) m
    f g : CategoryTheory.ComposableArrows C m
    app : CategoryTheory.Iso (f.obj' n ⋯) (g.obj' n ⋯)
    app' : CategoryTheory.Iso (f.obj' (HAdd.hAdd n 1) h) (g.obj' (HAdd.hAdd n 1) h)
    w : Eq (CategoryTheory.CategoryStruct.comp (f.map' n (HAdd.hAdd n 1) ⋯ h) app' …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (g.map' n (HAdd.hAdd n 1) ⋯ h) app'.i …
  -/
  rw [← cancel_epi app.hom, ← reassoc_of% w, app'.hom_inv_id, comp_id, app.hom_inv_id_assoc]
  /-
    🎉 no goals
  -/


/-- Constructor for isomorphisms in `ComposableArrows C 4`. -/
@[simps]
def isoMk₄ {f g : ComposableArrows C 4}
            /-
              C : Type u_1
              inst✝ : CategoryTheory.Category.{?u.315694, u_1} C
              n m : Nat
              F G : CategoryTheory.ComposableArrows C n
              f g : CategoryTheory.ComposableArrows C 4
              ⊢ LE.le 0 4
            -/
            /-
              🎉 no goals
            -/
                       /-
                         🎉 no goals
                       -/
                                         /-
                                           🎉 no goals
                                         -/
                                                    /-
                                                      🎉 no goals
                                                    -/
                                                                      /-
                                                                        🎉 no goals
                                                                      -/
    (app₀ : f.obj' 0 ≅ g.obj' 0) (app₁ : f.obj' 1 ≅ g.obj' 1) (app₂ : f.obj' 2 ≅ g.obj' 2)
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/
            /-
              C : Type u_1
              inst✝ : CategoryTheory.Category.{?u.315694, u_1} C
              n m : Nat
              F G : CategoryTheory.ComposableArrows C n
              f g : CategoryTheory.ComposableArrows C 4
              app₀ : CategoryTheory.Iso (f.obj' 0 ⋯) (g.obj' 0 ⋯)
              app₁ : CategoryTheory.Iso (f.obj' 1 ⋯) (g.obj' 1 ⋯)
              app₂ : CategoryTheory.Iso (f.obj' 2 ⋯) (g.obj' 2 ⋯)
              ⊢ LE.le 3 4
            -/
            /-
              🎉 no goals
            -/
                       /-
                         🎉 no goals
                       -/
                                         /-
                                           🎉 no goals
                                         -/
    (app₃ : f.obj' 3 ≅ g.obj' 3) (app₄ : f.obj' 4 ≅ g.obj' 4)
                                                    /-
                                                      🎉 no goals
                                                    -/
          /-
            C : Type u_1
            inst✝ : CategoryTheory.Category.{?u.315694, u_1} C
            n m : Nat
            F G : CategoryTheory.ComposableArrows C n
            f g : CategoryTheory.ComposableArrows C 4
            app₀ : CategoryTheory.Iso (f.obj' 0 ⋯) (g.obj' 0 ⋯)
            app₁ : CategoryTheory.Iso (f.obj' 1 ⋯) (g.obj' 1 ⋯)
            app₂ : CategoryTheory.Iso (f.obj' 2 ⋯) (g.obj' 2 ⋯)
            app₃ : CategoryTheory.Iso (f.obj' 3 ⋯) (g.obj' 3 ⋯)
            app₄ : CategoryTheory.Iso (f.obj' 4 ⋯) (g.obj' 4 ⋯)
            ⊢ LE.le 0 1
          -/
          /-
            🎉 no goals
          -/
          /-
            🎉 no goals
          -/
                                             /-
                                               🎉 no goals
                                             -/
    (w₀ : f.map' 0 1 ≫ app₁.hom = app₀.hom ≫ g.map' 0 1)
                                             /-
                                               🎉 no goals
                                             -/
          /-
            C : Type u_1
            inst✝ : CategoryTheory.Category.{?u.315694, u_1} C
            n m : Nat
            F G : CategoryTheory.ComposableArrows C n
            f g : CategoryTheory.ComposableArrows C 4
            app₀ : CategoryTheory.Iso (f.obj' 0 ⋯) (g.obj' 0 ⋯)
            app₁ : CategoryTheory.Iso (f.obj' 1 ⋯) (g.obj' 1 ⋯)
            app₂ : CategoryTheory.Iso (f.obj' 2 ⋯) (g.obj' 2 ⋯)
            app₃ : CategoryTheory.Iso (f.obj' 3 ⋯) (g.obj' 3 ⋯)
            app₄ : CategoryTheory.Iso (f.obj' 4 ⋯) (g.obj' 4 ⋯)
            w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁.hom) (Catego …
            ⊢ LE.le 1 2
          -/
          /-
            🎉 no goals
          -/
          /-
            🎉 no goals
          -/
                                             /-
                                               🎉 no goals
                                             -/
    (w₁ : f.map' 1 2 ≫ app₂.hom = app₁.hom ≫ g.map' 1 2)
                                             /-
                                               🎉 no goals
                                             -/
          /-
            C : Type u_1
            inst✝ : CategoryTheory.Category.{?u.315694, u_1} C
            n m : Nat
            F G : CategoryTheory.ComposableArrows C n
            f g : CategoryTheory.ComposableArrows C 4
            app₀ : CategoryTheory.Iso (f.obj' 0 ⋯) (g.obj' 0 ⋯)
            app₁ : CategoryTheory.Iso (f.obj' 1 ⋯) (g.obj' 1 ⋯)
            app₂ : CategoryTheory.Iso (f.obj' 2 ⋯) (g.obj' 2 ⋯)
            app₃ : CategoryTheory.Iso (f.obj' 3 ⋯) (g.obj' 3 ⋯)
            app₄ : CategoryTheory.Iso (f.obj' 4 ⋯) (g.obj' 4 ⋯)
            w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁.hom) (Catego …
            w₁ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 1 2 ⋯ ⋯) app₂.hom) (Catego …
            ⊢ LE.le 2 3
          -/
          /-
            🎉 no goals
          -/
          /-
            🎉 no goals
          -/
                                             /-
                                               🎉 no goals
                                             -/
    (w₂ : f.map' 2 3 ≫ app₃.hom = app₂.hom ≫ g.map' 2 3)
                                             /-
                                               🎉 no goals
                                             -/
          /-
            C : Type u_1
            inst✝ : CategoryTheory.Category.{?u.315694, u_1} C
            n m : Nat
            F G : CategoryTheory.ComposableArrows C n
            f g : CategoryTheory.ComposableArrows C 4
            app₀ : CategoryTheory.Iso (f.obj' 0 ⋯) (g.obj' 0 ⋯)
            app₁ : CategoryTheory.Iso (f.obj' 1 ⋯) (g.obj' 1 ⋯)
            app₂ : CategoryTheory.Iso (f.obj' 2 ⋯) (g.obj' 2 ⋯)
            app₃ : CategoryTheory.Iso (f.obj' 3 ⋯) (g.obj' 3 ⋯)
            app₄ : CategoryTheory.Iso (f.obj' 4 ⋯) (g.obj' 4 ⋯)
            w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁.hom) (Catego …
            w₁ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 1 2 ⋯ ⋯) app₂.hom) (Catego …
            w₂ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 2 3 ⋯ ⋯) app₃.hom) (Catego …
            ⊢ LE.le 3 4
          -/
          /-
            🎉 no goals
          -/
          /-
            🎉 no goals
          -/
                                             /-
                                               🎉 no goals
                                             -/
    (w₃ : f.map' 3 4 ≫ app₄.hom = app₃.hom ≫ g.map' 3 4) :
                                             /-
                                               🎉 no goals
                                             -/
    f ≅ g where
  hom := homMk₄ app₀.hom app₁.hom app₂.hom app₃.hom app₄.hom w₀ w₁ w₂ w₃
  inv := homMk₄ app₀.inv app₁.inv app₂.inv app₃.inv app₄.inv
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.315694, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 4
          app₀ : CategoryTheory.Iso (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : CategoryTheory.Iso (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : CategoryTheory.Iso (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : CategoryTheory.Iso (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          app₄ : CategoryTheory.Iso (f.obj' 4 ⋯) (g.obj' 4 ⋯)
          w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁.hom) (Catego …
          w₁ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 1 2 ⋯ ⋯) app₂.hom) (Catego …
          w₂ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 2 3 ⋯ ⋯) app₃.hom) (Catego …
          w₃ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 3 4 ⋯ ⋯) app₄.hom) (Catego …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (g.map' 0 1 ⋯ ⋯) app₁.inv) (CategoryT …
        -/
    (by rw [map'_inv_eq_inv_map' (by valid) app₀ app₁ w₀])
        /-
          🎉 no goals
        -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.315694, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 4
          app₀ : CategoryTheory.Iso (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : CategoryTheory.Iso (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : CategoryTheory.Iso (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : CategoryTheory.Iso (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          app₄ : CategoryTheory.Iso (f.obj' 4 ⋯) (g.obj' 4 ⋯)
          w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁.hom) (Catego …
          w₁ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 1 2 ⋯ ⋯) app₂.hom) (Catego …
          w₂ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 2 3 ⋯ ⋯) app₃.hom) (Catego …
          w₃ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 3 4 ⋯ ⋯) app₄.hom) (Catego …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (g.map' 1 2 ⋯ ⋯) app₂.inv) (CategoryT …
        -/
    (by rw [map'_inv_eq_inv_map' (by valid) app₁ app₂ w₁])
        /-
          🎉 no goals
        -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.315694, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 4
          app₀ : CategoryTheory.Iso (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : CategoryTheory.Iso (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : CategoryTheory.Iso (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : CategoryTheory.Iso (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          app₄ : CategoryTheory.Iso (f.obj' 4 ⋯) (g.obj' 4 ⋯)
          w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁.hom) (Catego …
          w₁ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 1 2 ⋯ ⋯) app₂.hom) (Catego …
          w₂ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 2 3 ⋯ ⋯) app₃.hom) (Catego …
          w₃ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 3 4 ⋯ ⋯) app₄.hom) (Catego …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (g.map' 2 3 ⋯ ⋯) app₃.inv) (CategoryT …
        -/
    (by rw [map'_inv_eq_inv_map' (by valid) app₂ app₃ w₂])
        /-
          🎉 no goals
        -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.315694, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 4
          app₀ : CategoryTheory.Iso (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : CategoryTheory.Iso (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : CategoryTheory.Iso (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : CategoryTheory.Iso (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          app₄ : CategoryTheory.Iso (f.obj' 4 ⋯) (g.obj' 4 ⋯)
          w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁.hom) (Catego …
          w₁ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 1 2 ⋯ ⋯) app₂.hom) (Catego …
          w₂ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 2 3 ⋯ ⋯) app₃.hom) (Catego …
          w₃ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 3 4 ⋯ ⋯) app₄.hom) (Catego …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (g.map' 3 4 ⋯ ⋯) app₄.inv) (CategoryT …
        -/
    (by rw [map'_inv_eq_inv_map' (by valid) app₃ app₄ w₃])
        /-
          🎉 no goals
        -/


lemma ext₄ {f g : ComposableArrows C 4}
          /-
            C : Type u_1
            inst✝ : CategoryTheory.Category.{?u.339191, u_1} C
            n m : Nat
            F G : CategoryTheory.ComposableArrows C n
            f g : CategoryTheory.ComposableArrows C 4
            ⊢ LE.le 0 4
          -/
          /-
            🎉 no goals
          -/
                     /-
                       🎉 no goals
                     -/
                                     /-
                                       🎉 no goals
                                     -/
                                                /-
                                                  🎉 no goals
                                                -/
                                                                /-
                                                                  🎉 no goals
                                                                -/
    (h₀ : f.obj' 0 = g.obj' 0) (h₁ : f.obj' 1 = g.obj' 1) (h₂ : f.obj' 2 = g.obj' 2)
                                                                           /-
                                                                             🎉 no goals
                                                                           -/
          /-
            C : Type u_1
            inst✝ : CategoryTheory.Category.{?u.339191, u_1} C
            n m : Nat
            F G : CategoryTheory.ComposableArrows C n
            f g : CategoryTheory.ComposableArrows C 4
            h₀ : Eq (f.obj' 0 ⋯) (g.obj' 0 ⋯)
            h₁ : Eq (f.obj' 1 ⋯) (g.obj' 1 ⋯)
            h₂ : Eq (f.obj' 2 ⋯) (g.obj' 2 ⋯)
            ⊢ LE.le 3 4
          -/
          /-
            🎉 no goals
          -/
                     /-
                       🎉 no goals
                     -/
                                     /-
                                       🎉 no goals
                                     -/
    (h₃ : f.obj' 3 = g.obj' 3) (h₄ : f.obj' 4 = g.obj' 4)
                                                /-
                                                  🎉 no goals
                                                -/
          /-
            C : Type u_1
            inst✝ : CategoryTheory.Category.{?u.339191, u_1} C
            n m : Nat
            F G : CategoryTheory.ComposableArrows C n
            f g : CategoryTheory.ComposableArrows C 4
            h₀ : Eq (f.obj' 0 ⋯) (g.obj' 0 ⋯)
            h₁ : Eq (f.obj' 1 ⋯) (g.obj' 1 ⋯)
            h₂ : Eq (f.obj' 2 ⋯) (g.obj' 2 ⋯)
            h₃ : Eq (f.obj' 3 ⋯) (g.obj' 3 ⋯)
            h₄ : Eq (f.obj' 4 ⋯) (g.obj' 4 ⋯)
            ⊢ LE.le 0 1
          -/
          /-
            🎉 no goals
          -/
          /-
            🎉 no goals
          -/
                                    /-
                                      🎉 no goals
                                    -/
    (w₀ : f.map' 0 1 = eqToHom h₀ ≫ g.map' 0 1 ≫ eqToHom h₁.symm)
                                    /-
                                      🎉 no goals
                                    -/
          /-
            C : Type u_1
            inst✝ : CategoryTheory.Category.{?u.339191, u_1} C
            n m : Nat
            F G : CategoryTheory.ComposableArrows C n
            f g : CategoryTheory.ComposableArrows C 4
            h₀ : Eq (f.obj' 0 ⋯) (g.obj' 0 ⋯)
            h₁ : Eq (f.obj' 1 ⋯) (g.obj' 1 ⋯)
            h₂ : Eq (f.obj' 2 ⋯) (g.obj' 2 ⋯)
            h₃ : Eq (f.obj' 3 ⋯) (g.obj' 3 ⋯)
            h₄ : Eq (f.obj' 4 ⋯) (g.obj' 4 ⋯)
            w₀ : Eq (f.map' 0 1 ⋯ ⋯) (CategoryTheory.CategoryStruct.comp (CategoryTheory.e …
            ⊢ LE.le 1 2
          -/
          /-
            🎉 no goals
          -/
          /-
            🎉 no goals
          -/
                                    /-
                                      🎉 no goals
                                    -/
    (w₁ : f.map' 1 2 = eqToHom h₁ ≫ g.map' 1 2 ≫ eqToHom h₂.symm)
                                    /-
                                      🎉 no goals
                                    -/
          /-
            C : Type u_1
            inst✝ : CategoryTheory.Category.{?u.339191, u_1} C
            n m : Nat
            F G : CategoryTheory.ComposableArrows C n
            f g : CategoryTheory.ComposableArrows C 4
            h₀ : Eq (f.obj' 0 ⋯) (g.obj' 0 ⋯)
            h₁ : Eq (f.obj' 1 ⋯) (g.obj' 1 ⋯)
            h₂ : Eq (f.obj' 2 ⋯) (g.obj' 2 ⋯)
            h₃ : Eq (f.obj' 3 ⋯) (g.obj' 3 ⋯)
            h₄ : Eq (f.obj' 4 ⋯) (g.obj' 4 ⋯)
            w₀ : Eq (f.map' 0 1 ⋯ ⋯) (CategoryTheory.CategoryStruct.comp (CategoryTheory.e …
            w₁ : Eq (f.map' 1 2 ⋯ ⋯) (CategoryTheory.CategoryStruct.comp (CategoryTheory.e …
            ⊢ LE.le 2 3
          -/
          /-
            🎉 no goals
          -/
          /-
            🎉 no goals
          -/
                                    /-
                                      🎉 no goals
                                    -/
    (w₂ : f.map' 2 3 = eqToHom h₂ ≫ g.map' 2 3 ≫ eqToHom h₃.symm)
                                    /-
                                      🎉 no goals
                                    -/
          /-
            C : Type u_1
            inst✝ : CategoryTheory.Category.{?u.339191, u_1} C
            n m : Nat
            F G : CategoryTheory.ComposableArrows C n
            f g : CategoryTheory.ComposableArrows C 4
            h₀ : Eq (f.obj' 0 ⋯) (g.obj' 0 ⋯)
            h₁ : Eq (f.obj' 1 ⋯) (g.obj' 1 ⋯)
            h₂ : Eq (f.obj' 2 ⋯) (g.obj' 2 ⋯)
            h₃ : Eq (f.obj' 3 ⋯) (g.obj' 3 ⋯)
            h₄ : Eq (f.obj' 4 ⋯) (g.obj' 4 ⋯)
            w₀ : Eq (f.map' 0 1 ⋯ ⋯) (CategoryTheory.CategoryStruct.comp (CategoryTheory.e …
            w₁ : Eq (f.map' 1 2 ⋯ ⋯) (CategoryTheory.CategoryStruct.comp (CategoryTheory.e …
            w₂ : Eq (f.map' 2 3 ⋯ ⋯) (CategoryTheory.CategoryStruct.comp (CategoryTheory.e …
            ⊢ LE.le 3 4
          -/
          /-
            🎉 no goals
          -/
          /-
            🎉 no goals
          -/
                                    /-
                                      🎉 no goals
                                    -/
    (w₃ : f.map' 3 4 = eqToHom h₃ ≫ g.map' 3 4 ≫ eqToHom h₄.symm) :
                                    /-
                                      🎉 no goals
                                    -/
    f = g :=
  ext_succ h₀ (ext₃ h₁ h₂ h₃ h₄ w₁ w₂ w₃) w₀


lemma mk₄_surjective (X : ComposableArrows C 4) :
    ∃ (X₀ X₁ X₂ X₃ X₄ : C) (f₀ : X₀ ⟶ X₁) (f₁ : X₁ ⟶ X₂) (f₂ : X₂ ⟶ X₃) (f₃ : X₃ ⟶ X₄),
      X = mk₄ f₀ f₁ f₂ f₃ :=
                  /-
                    C : Type u_1
                    inst✝ : CategoryTheory.Category.{u_2, u_1} C
                    X : CategoryTheory.ComposableArrows C 4
                    ⊢ LE.le 0 1
                  -/
                  /-
                    🎉 no goals
                  -/
                  /-
                    🎉 no goals
                  -/
                              /-
                                🎉 no goals
                              -/
                              /-
                                🎉 no goals
                              -/
                                          /-
                                            🎉 no goals
                                          -/
                                          /-
                                            🎉 no goals
                                          -/
                                                      /-
                                                        🎉 no goals
                                                      -/
  ⟨_, _, _, _, _, X.map' 0 1, X.map' 1 2, X.map' 2 3, X.map' 3 4,
                                                      /-
                                                        🎉 no goals
                                                      -/
                                 /-
                                   C : Type u_1
                                   inst✝ : CategoryTheory.Category.{u_2, u_1} C
                                   X : CategoryTheory.ComposableArrows C 4
                                   ⊢ Eq (X.map' 0 1 ⋯ ⋯) (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqTo …
                                 -/
                                 /-
                                   🎉 no goals
                                 -/
                                           /-
                                             🎉 no goals
                                           -/
                                                     /-
                                                       🎉 no goals
                                                     -/
    ext₄ rfl rfl rfl rfl rfl (by simp) (by simp) (by simp) (by simp)⟩
                                                               /-
                                                                 🎉 no goals
                                                               -/


    🎉 no goals
                                       -/
                                                  /-
                                                    🎉 no goals
                                                  -/
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
  (app₀ : f.obj' 0 ⟶ g.obj' 0) (app₁ : f.obj' 1 ⟶ g.obj' 1) (app₂ : f.obj' 2 ⟶ g.obj' 2)
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/
          /-
            C : Type u_1
            inst✝ : CategoryTheory.Category.{?u.377079, u_1} C
            n m : Nat
            F G : CategoryTheory.ComposableArrows C n
            f g : CategoryTheory.ComposableArrows C 5
            app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
            app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
            app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
            ⊢ LE.le 3 5
          -/
          /-
            🎉 no goals
          -/
                     /-
                       🎉 no goals
                     -/
                                       /-
                                         🎉 no goals
                                       -/
                                                  /-
                                                    🎉 no goals
                                                  -/
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
  (app₃ : f.obj' 3 ⟶ g.obj' 3) (app₄ : f.obj' 4 ⟶ g.obj' 4) (app₅ : f.obj' 5 ⟶ g.obj' 5)
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.377079, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 5
          app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          app₄ : Quiver.Hom (f.obj' 4 ⋯) (g.obj' 4 ⋯)
          app₅ : Quiver.Hom (f.obj' 5 ⋯) (g.obj' 5 ⋯)
          ⊢ LE.le 0 1
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
                                   /-
                                     🎉 no goals
                                   -/
  (w₀ : f.map' 0 1 ≫ app₁ = app₀ ≫ g.map' 0 1)
                                   /-
                                     🎉 no goals
                                   -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.377079, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 5
          app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          app₄ : Quiver.Hom (f.obj' 4 ⋯) (g.obj' 4 ⋯)
          app₅ : Quiver.Hom (f.obj' 5 ⋯) (g.obj' 5 ⋯)
          w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁) (CategoryTh …
          ⊢ LE.le 1 2
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
                                   /-
                                     🎉 no goals
                                   -/
  (w₁ : f.map' 1 2 ≫ app₂ = app₁ ≫ g.map' 1 2)
                                   /-
                                     🎉 no goals
                                   -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.377079, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 5
          app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          app₄ : Quiver.Hom (f.obj' 4 ⋯) (g.obj' 4 ⋯)
          app₅ : Quiver.Hom (f.obj' 5 ⋯) (g.obj' 5 ⋯)
          w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁) (CategoryTh …
          w₁ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 1 2 ⋯ ⋯) app₂) (CategoryTh …
          ⊢ LE.le 2 3
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
                                   /-
                                     🎉 no goals
                                   -/
  (w₂ : f.map' 2 3 ≫ app₃ = app₂ ≫ g.map' 2 3)
                                   /-
                                     🎉 no goals
                                   -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.377079, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 5
          app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          app₄ : Quiver.Hom (f.obj' 4 ⋯) (g.obj' 4 ⋯)
          app₅ : Quiver.Hom (f.obj' 5 ⋯) (g.obj' 5 ⋯)
          w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁) (CategoryTh …
          w₁ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 1 2 ⋯ ⋯) app₂) (CategoryTh …
          w₂ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 2 3 ⋯ ⋯) app₃) (CategoryTh …
          ⊢ LE.le 3 4
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
                                   /-
                                     🎉 no goals
                                   -/
  (w₃ : f.map' 3 4 ≫ app₄ = app₃ ≫ g.map' 3 4)
                                   /-
                                     🎉 no goals
                                   -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.377079, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 5
          app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          app₄ : Quiver.Hom (f.obj' 4 ⋯) (g.obj' 4 ⋯)
          app₅ : Quiver.Hom (f.obj' 5 ⋯) (g.obj' 5 ⋯)
          w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁) (CategoryTh …
          w₁ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 1 2 ⋯ ⋯) app₂) (CategoryTh …
          w₂ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 2 3 ⋯ ⋯) app₃) (CategoryTh …
          w₃ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 3 4 ⋯ ⋯) app₄) (CategoryTh …
          ⊢ LE.le 4 5
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
                                   /-
                                     🎉 no goals
                                   -/
  (w₄ : f.map' 4 5 ≫ app₅ = app₄ ≫ g.map' 4 5)
                                   /-
                                     🎉 no goals
                                   -/

/-- Constructor for morphisms in `ComposableArrows C 5`. -/
def homMk₅ : f ⟶ g := homMkSucc app₀ (homMk₄ app₁ app₂ app₃ app₄ app₅ w₁ w₂ w₃ w₄) w₀


als
                                                  -/
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
  (app₀ : f.obj' 0 ⟶ g.obj' 0) (app₁ : f.obj' 1 ⟶ g.obj' 1) (app₂ : f.obj' 2 ⟶ g.obj' 2)
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/
          /-
            C : Type u_1
            inst✝ : CategoryTheory.Category.{?u.381858, u_1} C
            n m : Nat
            F G : CategoryTheory.ComposableArrows C n
            f g : CategoryTheory.ComposableArrows C 5
            app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
            app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
            app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
            ⊢ LE.le 3 5
          -/
          /-
            🎉 no goals
          -/
                     /-
                       🎉 no goals
                     -/
                                       /-
                                         🎉 no goals
                                       -/
                                                  /-
                                                    🎉 no goals
                                                  -/
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
  (app₃ : f.obj' 3 ⟶ g.obj' 3) (app₄ : f.obj' 4 ⟶ g.obj' 4) (app₅ : f.obj' 5 ⟶ g.obj' 5)
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.381858, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 5
          app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          app₄ : Quiver.Hom (f.obj' 4 ⋯) (g.obj' 4 ⋯)
          app₅ : Quiver.Hom (f.obj' 5 ⋯) (g.obj' 5 ⋯)
          ⊢ LE.le 0 1
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
                                   /-
                                     🎉 no goals
                                   -/
  (w₀ : f.map' 0 1 ≫ app₁ = app₀ ≫ g.map' 0 1)
                                   /-
                                     🎉 no goals
                                   -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.381858, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 5
          app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          app₄ : Quiver.Hom (f.obj' 4 ⋯) (g.obj' 4 ⋯)
          app₅ : Quiver.Hom (f.obj' 5 ⋯) (g.obj' 5 ⋯)
          w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁) (CategoryTh …
          ⊢ LE.le 1 2
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
                                   /-
                                     🎉 no goals
                                   -/
  (w₁ : f.map' 1 2 ≫ app₂ = app₁ ≫ g.map' 1 2)
                                   /-
                                     🎉 no goals
                                   -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.381858, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 5
          app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          app₄ : Quiver.Hom (f.obj' 4 ⋯) (g.obj' 4 ⋯)
          app₅ : Quiver.Hom (f.obj' 5 ⋯) (g.obj' 5 ⋯)
          w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁) (CategoryTh …
          w₁ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 1 2 ⋯ ⋯) app₂) (CategoryTh …
          ⊢ LE.le 2 3
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
                                   /-
                                     🎉 no goals
                                   -/
  (w₂ : f.map' 2 3 ≫ app₃ = app₂ ≫ g.map' 2 3)
                                   /-
                                     🎉 no goals
                                   -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.381858, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 5
          app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          app₄ : Quiver.Hom (f.obj' 4 ⋯) (g.obj' 4 ⋯)
          app₅ : Quiver.Hom (f.obj' 5 ⋯) (g.obj' 5 ⋯)
          w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁) (CategoryTh …
          w₁ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 1 2 ⋯ ⋯) app₂) (CategoryTh …
          w₂ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 2 3 ⋯ ⋯) app₃) (CategoryTh …
          ⊢ LE.le 3 4
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
                                   /-
                                     🎉 no goals
                                   -/
  (w₃ : f.map' 3 4 ≫ app₄ = app₃ ≫ g.map' 3 4)
                                   /-
                                     🎉 no goals
                                   -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.381858, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 5
          app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          app₄ : Quiver.Hom (f.obj' 4 ⋯) (g.obj' 4 ⋯)
          app₅ : Quiver.Hom (f.obj' 5 ⋯) (g.obj' 5 ⋯)
          w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁) (CategoryTh …
          w₁ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 1 2 ⋯ ⋯) app₂) (CategoryTh …
          w₂ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 2 3 ⋯ ⋯) app₃) (CategoryTh …
          w₃ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 3 4 ⋯ ⋯) app₄) (CategoryTh …
          ⊢ LE.le 4 5
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
                                   /-
                                     🎉 no goals
                                   -/
  (w₄ : f.map' 4 5 ≫ app₅ = app₄ ≫ g.map' 4 5)
                                   /-
                                     🎉 no goals
                                   -/

/-- Constructor for morphisms in `ComposableArrows C 5`. -/
def homMk₅ : f ⟶ g := homMkSucc app₀ (homMk₄ app₁ app₂ app₃ app₄ app₅ w₁ w₂ w₃ w₄) w₀

@[simp]
lemma homMk₅_app_zero : (homMk₅ app₀ app₁ app₂ app₃ app₄ app₅ w₀ w₁ w₂ w₃ w₄).app 0 = app₀ := rfl


                                                               🎉 no goals
                                                                    -/
  (app₀ : f.obj' 0 ⟶ g.obj' 0) (app₁ : f.obj' 1 ⟶ g.obj' 1) (app₂ : f.obj' 2 ⟶ g.obj' 2)
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/
          /-
            C : Type u_1
            inst✝ : CategoryTheory.Category.{?u.386392, u_1} C
            n m : Nat
            F G : CategoryTheory.ComposableArrows C n
            f g : CategoryTheory.ComposableArrows C 5
            app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
            app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
            app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
            ⊢ LE.le 3 5
          -/
          /-
            🎉 no goals
          -/
                     /-
                       🎉 no goals
                     -/
                                       /-
                                         🎉 no goals
                                       -/
                                                  /-
                                                    🎉 no goals
                                                  -/
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
  (app₃ : f.obj' 3 ⟶ g.obj' 3) (app₄ : f.obj' 4 ⟶ g.obj' 4) (app₅ : f.obj' 5 ⟶ g.obj' 5)
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.386392, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 5
          app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          app₄ : Quiver.Hom (f.obj' 4 ⋯) (g.obj' 4 ⋯)
          app₅ : Quiver.Hom (f.obj' 5 ⋯) (g.obj' 5 ⋯)
          ⊢ LE.le 0 1
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
                                   /-
                                     🎉 no goals
                                   -/
  (w₀ : f.map' 0 1 ≫ app₁ = app₀ ≫ g.map' 0 1)
                                   /-
                                     🎉 no goals
                                   -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.386392, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 5
          app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          app₄ : Quiver.Hom (f.obj' 4 ⋯) (g.obj' 4 ⋯)
          app₅ : Quiver.Hom (f.obj' 5 ⋯) (g.obj' 5 ⋯)
          w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁) (CategoryTh …
          ⊢ LE.le 1 2
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
                                   /-
                                     🎉 no goals
                                   -/
  (w₁ : f.map' 1 2 ≫ app₂ = app₁ ≫ g.map' 1 2)
                                   /-
                                     🎉 no goals
                                   -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.386392, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 5
          app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          app₄ : Quiver.Hom (f.obj' 4 ⋯) (g.obj' 4 ⋯)
          app₅ : Quiver.Hom (f.obj' 5 ⋯) (g.obj' 5 ⋯)
          w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁) (CategoryTh …
          w₁ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 1 2 ⋯ ⋯) app₂) (CategoryTh …
          ⊢ LE.le 2 3
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
                                   /-
                                     🎉 no goals
                                   -/
  (w₂ : f.map' 2 3 ≫ app₃ = app₂ ≫ g.map' 2 3)
                                   /-
                                     🎉 no goals
                                   -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.386392, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 5
          app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          app₄ : Quiver.Hom (f.obj' 4 ⋯) (g.obj' 4 ⋯)
          app₅ : Quiver.Hom (f.obj' 5 ⋯) (g.obj' 5 ⋯)
          w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁) (CategoryTh …
          w₁ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 1 2 ⋯ ⋯) app₂) (CategoryTh …
          w₂ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 2 3 ⋯ ⋯) app₃) (CategoryTh …
          ⊢ LE.le 3 4
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
                                   /-
                                     🎉 no goals
                                   -/
  (w₃ : f.map' 3 4 ≫ app₄ = app₃ ≫ g.map' 3 4)
                                   /-
                                     🎉 no goals
                                   -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.386392, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 5
          app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          app₄ : Quiver.Hom (f.obj' 4 ⋯) (g.obj' 4 ⋯)
          app₅ : Quiver.Hom (f.obj' 5 ⋯) (g.obj' 5 ⋯)
          w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁) (CategoryTh …
          w₁ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 1 2 ⋯ ⋯) app₂) (CategoryTh …
          w₂ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 2 3 ⋯ ⋯) app₃) (CategoryTh …
          w₃ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 3 4 ⋯ ⋯) app₄) (CategoryTh …
          ⊢ LE.le 4 5
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
                                   /-
                                     🎉 no goals
                                   -/
  (w₄ : f.map' 4 5 ≫ app₅ = app₄ ≫ g.map' 4 5)
                                   /-
                                     🎉 no goals
                                   -/

/-- Constructor for morphisms in `ComposableArrows C 5`. -/
def homMk₅ : f ⟶ g := homMkSucc app₀ (homMk₄ app₁ app₂ app₃ app₄ app₅ w₁ w₂ w₃ w₄) w₀

@[simp]
lemma homMk₅_app_zero : (homMk₅ app₀ app₁ app₂ app₃ app₄ app₅ w₀ w₁ w₂ w₃ w₄).app 0 = app₀ := rfl

@[simp]
lemma homMk₅_app_one : (homMk₅ app₀ app₁ app₂ app₃ app₄ app₅ w₀ w₁ w₂ w₃ w₄).app 1 = app₁ := rfl


           -/
  (app₀ : f.obj' 0 ⟶ g.obj' 0) (app₁ : f.obj' 1 ⟶ g.obj' 1) (app₂ : f.obj' 2 ⟶ g.obj' 2)
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/
          /-
            C : Type u_1
            inst✝ : CategoryTheory.Category.{?u.390936, u_1} C
            n m : Nat
            F G : CategoryTheory.ComposableArrows C n
            f g : CategoryTheory.ComposableArrows C 5
            app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
            app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
            app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
            ⊢ LE.le 3 5
          -/
          /-
            🎉 no goals
          -/
                     /-
                       🎉 no goals
                     -/
                                       /-
                                         🎉 no goals
                                       -/
                                                  /-
                                                    🎉 no goals
                                                  -/
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
  (app₃ : f.obj' 3 ⟶ g.obj' 3) (app₄ : f.obj' 4 ⟶ g.obj' 4) (app₅ : f.obj' 5 ⟶ g.obj' 5)
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.390936, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 5
          app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          app₄ : Quiver.Hom (f.obj' 4 ⋯) (g.obj' 4 ⋯)
          app₅ : Quiver.Hom (f.obj' 5 ⋯) (g.obj' 5 ⋯)
          ⊢ LE.le 0 1
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
                                   /-
                                     🎉 no goals
                                   -/
  (w₀ : f.map' 0 1 ≫ app₁ = app₀ ≫ g.map' 0 1)
                                   /-
                                     🎉 no goals
                                   -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.390936, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 5
          app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          app₄ : Quiver.Hom (f.obj' 4 ⋯) (g.obj' 4 ⋯)
          app₅ : Quiver.Hom (f.obj' 5 ⋯) (g.obj' 5 ⋯)
          w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁) (CategoryTh …
          ⊢ LE.le 1 2
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
                                   /-
                                     🎉 no goals
                                   -/
  (w₁ : f.map' 1 2 ≫ app₂ = app₁ ≫ g.map' 1 2)
                                   /-
                                     🎉 no goals
                                   -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.390936, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 5
          app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          app₄ : Quiver.Hom (f.obj' 4 ⋯) (g.obj' 4 ⋯)
          app₅ : Quiver.Hom (f.obj' 5 ⋯) (g.obj' 5 ⋯)
          w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁) (CategoryTh …
          w₁ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 1 2 ⋯ ⋯) app₂) (CategoryTh …
          ⊢ LE.le 2 3
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
                                   /-
                                     🎉 no goals
                                   -/
  (w₂ : f.map' 2 3 ≫ app₃ = app₂ ≫ g.map' 2 3)
                                   /-
                                     🎉 no goals
                                   -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.390936, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 5
          app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          app₄ : Quiver.Hom (f.obj' 4 ⋯) (g.obj' 4 ⋯)
          app₅ : Quiver.Hom (f.obj' 5 ⋯) (g.obj' 5 ⋯)
          w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁) (CategoryTh …
          w₁ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 1 2 ⋯ ⋯) app₂) (CategoryTh …
          w₂ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 2 3 ⋯ ⋯) app₃) (CategoryTh …
          ⊢ LE.le 3 4
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
                                   /-
                                     🎉 no goals
                                   -/
  (w₃ : f.map' 3 4 ≫ app₄ = app₃ ≫ g.map' 3 4)
                                   /-
                                     🎉 no goals
                                   -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.390936, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 5
          app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          app₄ : Quiver.Hom (f.obj' 4 ⋯) (g.obj' 4 ⋯)
          app₅ : Quiver.Hom (f.obj' 5 ⋯) (g.obj' 5 ⋯)
          w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁) (CategoryTh …
          w₁ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 1 2 ⋯ ⋯) app₂) (CategoryTh …
          w₂ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 2 3 ⋯ ⋯) app₃) (CategoryTh …
          w₃ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 3 4 ⋯ ⋯) app₄) (CategoryTh …
          ⊢ LE.le 4 5
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
                                   /-
                                     🎉 no goals
                                   -/
  (w₄ : f.map' 4 5 ≫ app₅ = app₄ ≫ g.map' 4 5)
                                   /-
                                     🎉 no goals
                                   -/

/-- Constructor for morphisms in `ComposableArrows C 5`. -/
def homMk₅ : f ⟶ g := homMkSucc app₀ (homMk₄ app₁ app₂ app₃ app₄ app₅ w₁ w₂ w₃ w₄) w₀

@[simp]
lemma homMk₅_app_zero : (homMk₅ app₀ app₁ app₂ app₃ app₄ app₅ w₀ w₁ w₂ w₃ w₄).app 0 = app₀ := rfl

@[simp]
lemma homMk₅_app_one : (homMk₅ app₀ app₁ app₂ app₃ app₄ app₅ w₀ w₁ w₂ w₃ w₄).app 1 = app₁ := rfl

@[simp]
lemma homMk₅_app_two :
                                                                     /-
                                                                       C : Type u_1
                                                                       inst✝ : CategoryTheory.Category.{?u.390936, u_1} C
                                                                       n m : Nat
                                                                       F G : CategoryTheory.ComposableArrows C n
                                                                       f g : CategoryTheory.ComposableArrows C 5
                                                                       app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
                                                                       app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
                                                                       app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
                                                                       app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
                                                                       app₄ : Quiver.Hom (f.obj' 4 ⋯) (g.obj' 4 ⋯)
                                                                       app₅ : Quiver.Hom (f.obj' 5 ⋯) (g.obj' 5 ⋯)
                                                                       w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁) (CategoryTh …
                                                                       w₁ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 1 2 ⋯ ⋯) app₂) (CategoryTh …
                                                                       w₂ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 2 3 ⋯ ⋯) app₃) (CategoryTh …
                                                                       w₃ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 3 4 ⋯ ⋯) app₄) (CategoryTh …
                                                                       w₄ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 4 5 ⋯ ⋯) app₅) (CategoryTh …
                                                                       ⊢ LT.lt 2 (HAdd.hAdd 5 1)
                                                                     -/
    (homMk₅ app₀ app₁ app₂ app₃ app₄ app₅ w₀ w₁ w₂ w₃ w₄).app ⟨2, by valid⟩ = app₂ := rfl
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


                                        /-
                                                                                 🎉 no goals
                                                                               -/
          /-
            C : Type u_1
            inst✝ : CategoryTheory.Category.{?u.395611, u_1} C
            n m : Nat
            F G : CategoryTheory.ComposableArrows C n
            f g : CategoryTheory.ComposableArrows C 5
            app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
            app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
            app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
            ⊢ LE.le 3 5
          -/
          /-
            🎉 no goals
          -/
                     /-
                       🎉 no goals
                     -/
                                       /-
                                         🎉 no goals
                                       -/
                                                  /-
                                                    🎉 no goals
                                                  -/
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
  (app₃ : f.obj' 3 ⟶ g.obj' 3) (app₄ : f.obj' 4 ⟶ g.obj' 4) (app₅ : f.obj' 5 ⟶ g.obj' 5)
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.395611, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 5
          app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          app₄ : Quiver.Hom (f.obj' 4 ⋯) (g.obj' 4 ⋯)
          app₅ : Quiver.Hom (f.obj' 5 ⋯) (g.obj' 5 ⋯)
          ⊢ LE.le 0 1
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
                                   /-
                                     🎉 no goals
                                   -/
  (w₀ : f.map' 0 1 ≫ app₁ = app₀ ≫ g.map' 0 1)
                                   /-
                                     🎉 no goals
                                   -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.395611, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 5
          app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          app₄ : Quiver.Hom (f.obj' 4 ⋯) (g.obj' 4 ⋯)
          app₅ : Quiver.Hom (f.obj' 5 ⋯) (g.obj' 5 ⋯)
          w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁) (CategoryTh …
          ⊢ LE.le 1 2
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
                                   /-
                                     🎉 no goals
                                   -/
  (w₁ : f.map' 1 2 ≫ app₂ = app₁ ≫ g.map' 1 2)
                                   /-
                                     🎉 no goals
                                   -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.395611, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 5
          app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          app₄ : Quiver.Hom (f.obj' 4 ⋯) (g.obj' 4 ⋯)
          app₅ : Quiver.Hom (f.obj' 5 ⋯) (g.obj' 5 ⋯)
          w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁) (CategoryTh …
          w₁ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 1 2 ⋯ ⋯) app₂) (CategoryTh …
          ⊢ LE.le 2 3
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
                                   /-
                                     🎉 no goals
                                   -/
  (w₂ : f.map' 2 3 ≫ app₃ = app₂ ≫ g.map' 2 3)
                                   /-
                                     🎉 no goals
                                   -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.395611, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 5
          app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          app₄ : Quiver.Hom (f.obj' 4 ⋯) (g.obj' 4 ⋯)
          app₅ : Quiver.Hom (f.obj' 5 ⋯) (g.obj' 5 ⋯)
          w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁) (CategoryTh …
          w₁ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 1 2 ⋯ ⋯) app₂) (CategoryTh …
          w₂ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 2 3 ⋯ ⋯) app₃) (CategoryTh …
          ⊢ LE.le 3 4
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
                                   /-
                                     🎉 no goals
                                   -/
  (w₃ : f.map' 3 4 ≫ app₄ = app₃ ≫ g.map' 3 4)
                                   /-
                                     🎉 no goals
                                   -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.395611, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 5
          app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          app₄ : Quiver.Hom (f.obj' 4 ⋯) (g.obj' 4 ⋯)
          app₅ : Quiver.Hom (f.obj' 5 ⋯) (g.obj' 5 ⋯)
          w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁) (CategoryTh …
          w₁ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 1 2 ⋯ ⋯) app₂) (CategoryTh …
          w₂ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 2 3 ⋯ ⋯) app₃) (CategoryTh …
          w₃ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 3 4 ⋯ ⋯) app₄) (CategoryTh …
          ⊢ LE.le 4 5
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
                                   /-
                                     🎉 no goals
                                   -/
  (w₄ : f.map' 4 5 ≫ app₅ = app₄ ≫ g.map' 4 5)
                                   /-
                                     🎉 no goals
                                   -/

/-- Constructor for morphisms in `ComposableArrows C 5`. -/
def homMk₅ : f ⟶ g := homMkSucc app₀ (homMk₄ app₁ app₂ app₃ app₄ app₅ w₁ w₂ w₃ w₄) w₀

@[simp]
lemma homMk₅_app_zero : (homMk₅ app₀ app₁ app₂ app₃ app₄ app₅ w₀ w₁ w₂ w₃ w₄).app 0 = app₀ := rfl

@[simp]
lemma homMk₅_app_one : (homMk₅ app₀ app₁ app₂ app₃ app₄ app₅ w₀ w₁ w₂ w₃ w₄).app 1 = app₁ := rfl

@[simp]
lemma homMk₅_app_two :
    (homMk₅ app₀ app₁ app₂ app₃ app₄ app₅ w₀ w₁ w₂ w₃ w₄).app ⟨2, by valid⟩ = app₂ := rfl

@[simp]
lemma homMk₅_app_three :
                                                                     /-
                                                                       C : Type u_1
                                                                       inst✝ : CategoryTheory.Category.{?u.395611, u_1} C
                                                                       n m : Nat
                                                                       F G : CategoryTheory.ComposableArrows C n
                                                                       f g : CategoryTheory.ComposableArrows C 5
                                                                       app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
                                                                       app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
                                                                       app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
                                                                       app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
                                                                       app₄ : Quiver.Hom (f.obj' 4 ⋯) (g.obj' 4 ⋯)
                                                                       app₅ : Quiver.Hom (f.obj' 5 ⋯) (g.obj' 5 ⋯)
                                                                       w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁) (CategoryTh …
                                                                       w₁ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 1 2 ⋯ ⋯) app₂) (CategoryTh …
                                                                       w₂ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 2 3 ⋯ ⋯) app₃) (CategoryTh …
                                                                       w₃ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 3 4 ⋯ ⋯) app₄) (CategoryTh …
                                                                       w₄ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 4 5 ⋯ ⋯) app₅) (CategoryTh …
                                                                       ⊢ LT.lt 3 (HAdd.hAdd 5 1)
                                                                     -/
    (homMk₅ app₀ app₁ app₂ app₃ app₄ app₅ w₀ w₁ w₂ w₃ w₄).app ⟨3, by valid⟩ = app₃ := rfl
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


                                                             -/
          /-
            C : Type u_1
            inst✝ : CategoryTheory.Category.{?u.400296, u_1} C
            n m : Nat
            F G : CategoryTheory.ComposableArrows C n
            f g : CategoryTheory.ComposableArrows C 5
            app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
            app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
            app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
            ⊢ LE.le 3 5
          -/
          /-
            🎉 no goals
          -/
                     /-
                       🎉 no goals
                     -/
                                       /-
                                         🎉 no goals
                                       -/
                                                  /-
                                                    🎉 no goals
                                                  -/
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
  (app₃ : f.obj' 3 ⟶ g.obj' 3) (app₄ : f.obj' 4 ⟶ g.obj' 4) (app₅ : f.obj' 5 ⟶ g.obj' 5)
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.400296, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 5
          app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          app₄ : Quiver.Hom (f.obj' 4 ⋯) (g.obj' 4 ⋯)
          app₅ : Quiver.Hom (f.obj' 5 ⋯) (g.obj' 5 ⋯)
          ⊢ LE.le 0 1
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
                                   /-
                                     🎉 no goals
                                   -/
  (w₀ : f.map' 0 1 ≫ app₁ = app₀ ≫ g.map' 0 1)
                                   /-
                                     🎉 no goals
                                   -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.400296, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 5
          app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          app₄ : Quiver.Hom (f.obj' 4 ⋯) (g.obj' 4 ⋯)
          app₅ : Quiver.Hom (f.obj' 5 ⋯) (g.obj' 5 ⋯)
          w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁) (CategoryTh …
          ⊢ LE.le 1 2
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
                                   /-
                                     🎉 no goals
                                   -/
  (w₁ : f.map' 1 2 ≫ app₂ = app₁ ≫ g.map' 1 2)
                                   /-
                                     🎉 no goals
                                   -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.400296, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 5
          app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          app₄ : Quiver.Hom (f.obj' 4 ⋯) (g.obj' 4 ⋯)
          app₅ : Quiver.Hom (f.obj' 5 ⋯) (g.obj' 5 ⋯)
          w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁) (CategoryTh …
          w₁ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 1 2 ⋯ ⋯) app₂) (CategoryTh …
          ⊢ LE.le 2 3
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
                                   /-
                                     🎉 no goals
                                   -/
  (w₂ : f.map' 2 3 ≫ app₃ = app₂ ≫ g.map' 2 3)
                                   /-
                                     🎉 no goals
                                   -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.400296, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 5
          app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          app₄ : Quiver.Hom (f.obj' 4 ⋯) (g.obj' 4 ⋯)
          app₅ : Quiver.Hom (f.obj' 5 ⋯) (g.obj' 5 ⋯)
          w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁) (CategoryTh …
          w₁ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 1 2 ⋯ ⋯) app₂) (CategoryTh …
          w₂ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 2 3 ⋯ ⋯) app₃) (CategoryTh …
          ⊢ LE.le 3 4
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
                                   /-
                                     🎉 no goals
                                   -/
  (w₃ : f.map' 3 4 ≫ app₄ = app₃ ≫ g.map' 3 4)
                                   /-
                                     🎉 no goals
                                   -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.400296, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 5
          app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          app₄ : Quiver.Hom (f.obj' 4 ⋯) (g.obj' 4 ⋯)
          app₅ : Quiver.Hom (f.obj' 5 ⋯) (g.obj' 5 ⋯)
          w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁) (CategoryTh …
          w₁ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 1 2 ⋯ ⋯) app₂) (CategoryTh …
          w₂ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 2 3 ⋯ ⋯) app₃) (CategoryTh …
          w₃ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 3 4 ⋯ ⋯) app₄) (CategoryTh …
          ⊢ LE.le 4 5
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
                                   /-
                                     🎉 no goals
                                   -/
  (w₄ : f.map' 4 5 ≫ app₅ = app₄ ≫ g.map' 4 5)
                                   /-
                                     🎉 no goals
                                   -/

/-- Constructor for morphisms in `ComposableArrows C 5`. -/
def homMk₅ : f ⟶ g := homMkSucc app₀ (homMk₄ app₁ app₂ app₃ app₄ app₅ w₁ w₂ w₃ w₄) w₀

@[simp]
lemma homMk₅_app_zero : (homMk₅ app₀ app₁ app₂ app₃ app₄ app₅ w₀ w₁ w₂ w₃ w₄).app 0 = app₀ := rfl

@[simp]
lemma homMk₅_app_one : (homMk₅ app₀ app₁ app₂ app₃ app₄ app₅ w₀ w₁ w₂ w₃ w₄).app 1 = app₁ := rfl

@[simp]
lemma homMk₅_app_two :
    (homMk₅ app₀ app₁ app₂ app₃ app₄ app₅ w₀ w₁ w₂ w₃ w₄).app ⟨2, by valid⟩ = app₂ := rfl

@[simp]
lemma homMk₅_app_three :
    (homMk₅ app₀ app₁ app₂ app₃ app₄ app₅ w₀ w₁ w₂ w₃ w₄).app ⟨3, by valid⟩ = app₃ := rfl

@[simp]
lemma homMk₅_app_four :
                                                                     /-
                                                                       C : Type u_1
                                                                       inst✝ : CategoryTheory.Category.{?u.400296, u_1} C
                                                                       n m : Nat
                                                                       F G : CategoryTheory.ComposableArrows C n
                                                                       f g : CategoryTheory.ComposableArrows C 5
                                                                       app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
                                                                       app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
                                                                       app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
                                                                       app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
                                                                       app₄ : Quiver.Hom (f.obj' 4 ⋯) (g.obj' 4 ⋯)
                                                                       app₅ : Quiver.Hom (f.obj' 5 ⋯) (g.obj' 5 ⋯)
                                                                       w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁) (CategoryTh …
                                                                       w₁ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 1 2 ⋯ ⋯) app₂) (CategoryTh …
                                                                       w₂ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 2 3 ⋯ ⋯) app₃) (CategoryTh …
                                                                       w₃ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 3 4 ⋯ ⋯) app₄) (CategoryTh …
                                                                       w₄ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 4 5 ⋯ ⋯) app₅) (CategoryTh …
                                                                       ⊢ LT.lt 4 (HAdd.hAdd 5 1)
                                                                     -/
    (homMk₅ app₀ app₁ app₂ app₃ app₄ app₅ w₀ w₁ w₂ w₃ w₄).app ⟨4, by valid⟩ = app₄ := rfl
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


993, u_1} C
            n m : Nat
            F G : CategoryTheory.ComposableArrows C n
            f g : CategoryTheory.ComposableArrows C 5
            app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
            app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
            app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
            ⊢ LE.le 3 5
          -/
          /-
            🎉 no goals
          -/
                     /-
                       🎉 no goals
                     -/
                                       /-
                                         🎉 no goals
                                       -/
                                                  /-
                                                    🎉 no goals
                                                  -/
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
  (app₃ : f.obj' 3 ⟶ g.obj' 3) (app₄ : f.obj' 4 ⟶ g.obj' 4) (app₅ : f.obj' 5 ⟶ g.obj' 5)
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.404993, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 5
          app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          app₄ : Quiver.Hom (f.obj' 4 ⋯) (g.obj' 4 ⋯)
          app₅ : Quiver.Hom (f.obj' 5 ⋯) (g.obj' 5 ⋯)
          ⊢ LE.le 0 1
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
                                   /-
                                     🎉 no goals
                                   -/
  (w₀ : f.map' 0 1 ≫ app₁ = app₀ ≫ g.map' 0 1)
                                   /-
                                     🎉 no goals
                                   -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.404993, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 5
          app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          app₄ : Quiver.Hom (f.obj' 4 ⋯) (g.obj' 4 ⋯)
          app₅ : Quiver.Hom (f.obj' 5 ⋯) (g.obj' 5 ⋯)
          w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁) (CategoryTh …
          ⊢ LE.le 1 2
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
                                   /-
                                     🎉 no goals
                                   -/
  (w₁ : f.map' 1 2 ≫ app₂ = app₁ ≫ g.map' 1 2)
                                   /-
                                     🎉 no goals
                                   -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.404993, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 5
          app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          app₄ : Quiver.Hom (f.obj' 4 ⋯) (g.obj' 4 ⋯)
          app₅ : Quiver.Hom (f.obj' 5 ⋯) (g.obj' 5 ⋯)
          w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁) (CategoryTh …
          w₁ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 1 2 ⋯ ⋯) app₂) (CategoryTh …
          ⊢ LE.le 2 3
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
                                   /-
                                     🎉 no goals
                                   -/
  (w₂ : f.map' 2 3 ≫ app₃ = app₂ ≫ g.map' 2 3)
                                   /-
                                     🎉 no goals
                                   -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.404993, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 5
          app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          app₄ : Quiver.Hom (f.obj' 4 ⋯) (g.obj' 4 ⋯)
          app₅ : Quiver.Hom (f.obj' 5 ⋯) (g.obj' 5 ⋯)
          w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁) (CategoryTh …
          w₁ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 1 2 ⋯ ⋯) app₂) (CategoryTh …
          w₂ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 2 3 ⋯ ⋯) app₃) (CategoryTh …
          ⊢ LE.le 3 4
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
                                   /-
                                     🎉 no goals
                                   -/
  (w₃ : f.map' 3 4 ≫ app₄ = app₃ ≫ g.map' 3 4)
                                   /-
                                     🎉 no goals
                                   -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.404993, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 5
          app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          app₄ : Quiver.Hom (f.obj' 4 ⋯) (g.obj' 4 ⋯)
          app₅ : Quiver.Hom (f.obj' 5 ⋯) (g.obj' 5 ⋯)
          w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁) (CategoryTh …
          w₁ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 1 2 ⋯ ⋯) app₂) (CategoryTh …
          w₂ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 2 3 ⋯ ⋯) app₃) (CategoryTh …
          w₃ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 3 4 ⋯ ⋯) app₄) (CategoryTh …
          ⊢ LE.le 4 5
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
                                   /-
                                     🎉 no goals
                                   -/
  (w₄ : f.map' 4 5 ≫ app₅ = app₄ ≫ g.map' 4 5)
                                   /-
                                     🎉 no goals
                                   -/

/-- Constructor for morphisms in `ComposableArrows C 5`. -/
def homMk₅ : f ⟶ g := homMkSucc app₀ (homMk₄ app₁ app₂ app₃ app₄ app₅ w₁ w₂ w₃ w₄) w₀

@[simp]
lemma homMk₅_app_zero : (homMk₅ app₀ app₁ app₂ app₃ app₄ app₅ w₀ w₁ w₂ w₃ w₄).app 0 = app₀ := rfl

@[simp]
lemma homMk₅_app_one : (homMk₅ app₀ app₁ app₂ app₃ app₄ app₅ w₀ w₁ w₂ w₃ w₄).app 1 = app₁ := rfl

@[simp]
lemma homMk₅_app_two :
    (homMk₅ app₀ app₁ app₂ app₃ app₄ app₅ w₀ w₁ w₂ w₃ w₄).app ⟨2, by valid⟩ = app₂ := rfl

@[simp]
lemma homMk₅_app_three :
    (homMk₅ app₀ app₁ app₂ app₃ app₄ app₅ w₀ w₁ w₂ w₃ w₄).app ⟨3, by valid⟩ = app₃ := rfl

@[simp]
lemma homMk₅_app_four :
    (homMk₅ app₀ app₁ app₂ app₃ app₄ app₅ w₀ w₁ w₂ w₃ w₄).app ⟨4, by valid⟩ = app₄ := rfl

@[simp]
lemma homMk₅_app_five :
                                                                     /-
                                                                       C : Type u_1
                                                                       inst✝ : CategoryTheory.Category.{?u.404993, u_1} C
                                                                       n m : Nat
                                                                       F G : CategoryTheory.ComposableArrows C n
                                                                       f g : CategoryTheory.ComposableArrows C 5
                                                                       app₀ : Quiver.Hom (f.obj' 0 ⋯) (g.obj' 0 ⋯)
                                                                       app₁ : Quiver.Hom (f.obj' 1 ⋯) (g.obj' 1 ⋯)
                                                                       app₂ : Quiver.Hom (f.obj' 2 ⋯) (g.obj' 2 ⋯)
                                                                       app₃ : Quiver.Hom (f.obj' 3 ⋯) (g.obj' 3 ⋯)
                                                                       app₄ : Quiver.Hom (f.obj' 4 ⋯) (g.obj' 4 ⋯)
                                                                       app₅ : Quiver.Hom (f.obj' 5 ⋯) (g.obj' 5 ⋯)
                                                                       w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁) (CategoryTh …
                                                                       w₁ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 1 2 ⋯ ⋯) app₂) (CategoryTh …
                                                                       w₂ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 2 3 ⋯ ⋯) app₃) (CategoryTh …
                                                                       w₃ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 3 4 ⋯ ⋯) app₄) (CategoryTh …
                                                                       w₄ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 4 5 ⋯ ⋯) app₅) (CategoryTh …
                                                                       ⊢ LT.lt 5 (HAdd.hAdd 5 1)
                                                                     -/
    (homMk₅ app₀ app₁ app₂ app₃ app₄ app₅ w₀ w₁ w₂ w₃ w₄).app ⟨5, by valid⟩ = app₅ := rfl
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


@[ext]
lemma hom_ext₅ {f g : ComposableArrows C 5} {φ φ' : f ⟶ g}
          /-
            C : Type u_1
            inst✝ : CategoryTheory.Category.{?u.409690, u_1} C
            n m : Nat
            F G : CategoryTheory.ComposableArrows C n
            f g : CategoryTheory.ComposableArrows C 5
            φ φ' : Quiver.Hom f g
            ⊢ LE.le 0 5
          -/
          /-
            🎉 no goals
          -/
                     /-
                       🎉 no goals
                     -/
                                      /-
                                        🎉 no goals
                                      -/
                                                 /-
                                                   🎉 no goals
                                                 -/
                                                                  /-
                                                                    🎉 no goals
                                                                  -/
    (h₀ : app' φ 0 = app' φ' 0) (h₁ : app' φ 1 = app' φ' 1) (h₂ : app' φ 2 = app' φ' 2)
                                                                             /-
                                                                               🎉 no goals
                                                                             -/
          /-
            C : Type u_1
            inst✝ : CategoryTheory.Category.{?u.409690, u_1} C
            n m : Nat
            F G : CategoryTheory.ComposableArrows C n
            f g : CategoryTheory.ComposableArrows C 5
            φ φ' : Quiver.Hom f g
            h₀ : Eq (CategoryTheory.ComposableArrows.app' φ 0 ⋯) (CategoryTheory.Composabl …
            h₁ : Eq (CategoryTheory.ComposableArrows.app' φ 1 ⋯) (CategoryTheory.Composabl …
            h₂ : Eq (CategoryTheory.ComposableArrows.app' φ 2 ⋯) (CategoryTheory.Composabl …
            ⊢ LE.le 3 5
          -/
          /-
            🎉 no goals
          -/
                     /-
                       🎉 no goals
                     -/
                                      /-
                                        🎉 no goals
                                      -/
                                                 /-
                                                   🎉 no goals
                                                 -/
                                                                  /-
                                                                    🎉 no goals
                                                                  -/
    (h₃ : app' φ 3 = app' φ' 3) (h₄ : app' φ 4 = app' φ' 4) (h₅ : app' φ 5 = app' φ' 5) :
                                                                             /-
                                                                               🎉 no goals
                                                                             -/
    φ = φ' :=
  hom_ext_succ h₀ (hom_ext₄ h₁ h₂ h₃ h₄ h₅)


/-- Constructor for isomorphisms in `ComposableArrows C 5`. -/
@[simps]
def isoMk₅ {f g : ComposableArrows C 5}
            /-
              C : Type u_1
              inst✝ : CategoryTheory.Category.{?u.412135, u_1} C
              n m : Nat
              F G : CategoryTheory.ComposableArrows C n
              f g : CategoryTheory.ComposableArrows C 5
              ⊢ LE.le 0 5
            -/
            /-
              🎉 no goals
            -/
                       /-
                         🎉 no goals
                       -/
                                         /-
                                           🎉 no goals
                                         -/
                                                    /-
                                                      🎉 no goals
                                                    -/
                                                                      /-
                                                                        🎉 no goals
                                                                      -/
    (app₀ : f.obj' 0 ≅ g.obj' 0) (app₁ : f.obj' 1 ≅ g.obj' 1) (app₂ : f.obj' 2 ≅ g.obj' 2)
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/
            /-
              C : Type u_1
              inst✝ : CategoryTheory.Category.{?u.412135, u_1} C
              n m : Nat
              F G : CategoryTheory.ComposableArrows C n
              f g : CategoryTheory.ComposableArrows C 5
              app₀ : CategoryTheory.Iso (f.obj' 0 ⋯) (g.obj' 0 ⋯)
              app₁ : CategoryTheory.Iso (f.obj' 1 ⋯) (g.obj' 1 ⋯)
              app₂ : CategoryTheory.Iso (f.obj' 2 ⋯) (g.obj' 2 ⋯)
              ⊢ LE.le 3 5
            -/
            /-
              🎉 no goals
            -/
                       /-
                         🎉 no goals
                       -/
                                         /-
                                           🎉 no goals
                                         -/
                                                    /-
                                                      🎉 no goals
                                                    -/
                                                                      /-
                                                                        🎉 no goals
                                                                      -/
    (app₃ : f.obj' 3 ≅ g.obj' 3) (app₄ : f.obj' 4 ≅ g.obj' 4) (app₅ : f.obj' 5 ≅ g.obj' 5)
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/
          /-
            C : Type u_1
            inst✝ : CategoryTheory.Category.{?u.412135, u_1} C
            n m : Nat
            F G : CategoryTheory.ComposableArrows C n
            f g : CategoryTheory.ComposableArrows C 5
            app₀ : CategoryTheory.Iso (f.obj' 0 ⋯) (g.obj' 0 ⋯)
            app₁ : CategoryTheory.Iso (f.obj' 1 ⋯) (g.obj' 1 ⋯)
            app₂ : CategoryTheory.Iso (f.obj' 2 ⋯) (g.obj' 2 ⋯)
            app₃ : CategoryTheory.Iso (f.obj' 3 ⋯) (g.obj' 3 ⋯)
            app₄ : CategoryTheory.Iso (f.obj' 4 ⋯) (g.obj' 4 ⋯)
            app₅ : CategoryTheory.Iso (f.obj' 5 ⋯) (g.obj' 5 ⋯)
            ⊢ LE.le 0 1
          -/
          /-
            🎉 no goals
          -/
          /-
            🎉 no goals
          -/
                                             /-
                                               🎉 no goals
                                             -/
    (w₀ : f.map' 0 1 ≫ app₁.hom = app₀.hom ≫ g.map' 0 1)
                                             /-
                                               🎉 no goals
                                             -/
          /-
            C : Type u_1
            inst✝ : CategoryTheory.Category.{?u.412135, u_1} C
            n m : Nat
            F G : CategoryTheory.ComposableArrows C n
            f g : CategoryTheory.ComposableArrows C 5
            app₀ : CategoryTheory.Iso (f.obj' 0 ⋯) (g.obj' 0 ⋯)
            app₁ : CategoryTheory.Iso (f.obj' 1 ⋯) (g.obj' 1 ⋯)
            app₂ : CategoryTheory.Iso (f.obj' 2 ⋯) (g.obj' 2 ⋯)
            app₃ : CategoryTheory.Iso (f.obj' 3 ⋯) (g.obj' 3 ⋯)
            app₄ : CategoryTheory.Iso (f.obj' 4 ⋯) (g.obj' 4 ⋯)
            app₅ : CategoryTheory.Iso (f.obj' 5 ⋯) (g.obj' 5 ⋯)
            w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁.hom) (Catego …
            ⊢ LE.le 1 2
          -/
          /-
            🎉 no goals
          -/
          /-
            🎉 no goals
          -/
                                             /-
                                               🎉 no goals
                                             -/
    (w₁ : f.map' 1 2 ≫ app₂.hom = app₁.hom ≫ g.map' 1 2)
                                             /-
                                               🎉 no goals
                                             -/
          /-
            C : Type u_1
            inst✝ : CategoryTheory.Category.{?u.412135, u_1} C
            n m : Nat
            F G : CategoryTheory.ComposableArrows C n
            f g : CategoryTheory.ComposableArrows C 5
            app₀ : CategoryTheory.Iso (f.obj' 0 ⋯) (g.obj' 0 ⋯)
            app₁ : CategoryTheory.Iso (f.obj' 1 ⋯) (g.obj' 1 ⋯)
            app₂ : CategoryTheory.Iso (f.obj' 2 ⋯) (g.obj' 2 ⋯)
            app₃ : CategoryTheory.Iso (f.obj' 3 ⋯) (g.obj' 3 ⋯)
            app₄ : CategoryTheory.Iso (f.obj' 4 ⋯) (g.obj' 4 ⋯)
            app₅ : CategoryTheory.Iso (f.obj' 5 ⋯) (g.obj' 5 ⋯)
            w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁.hom) (Catego …
            w₁ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 1 2 ⋯ ⋯) app₂.hom) (Catego …
            ⊢ LE.le 2 3
          -/
          /-
            🎉 no goals
          -/
          /-
            🎉 no goals
          -/
                                             /-
                                               🎉 no goals
                                             -/
    (w₂ : f.map' 2 3 ≫ app₃.hom = app₂.hom ≫ g.map' 2 3)
                                             /-
                                               🎉 no goals
                                             -/
          /-
            C : Type u_1
            inst✝ : CategoryTheory.Category.{?u.412135, u_1} C
            n m : Nat
            F G : CategoryTheory.ComposableArrows C n
            f g : CategoryTheory.ComposableArrows C 5
            app₀ : CategoryTheory.Iso (f.obj' 0 ⋯) (g.obj' 0 ⋯)
            app₁ : CategoryTheory.Iso (f.obj' 1 ⋯) (g.obj' 1 ⋯)
            app₂ : CategoryTheory.Iso (f.obj' 2 ⋯) (g.obj' 2 ⋯)
            app₃ : CategoryTheory.Iso (f.obj' 3 ⋯) (g.obj' 3 ⋯)
            app₄ : CategoryTheory.Iso (f.obj' 4 ⋯) (g.obj' 4 ⋯)
            app₅ : CategoryTheory.Iso (f.obj' 5 ⋯) (g.obj' 5 ⋯)
            w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁.hom) (Catego …
            w₁ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 1 2 ⋯ ⋯) app₂.hom) (Catego …
            w₂ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 2 3 ⋯ ⋯) app₃.hom) (Catego …
            ⊢ LE.le 3 4
          -/
          /-
            🎉 no goals
          -/
          /-
            🎉 no goals
          -/
                                             /-
                                               🎉 no goals
                                             -/
    (w₃ : f.map' 3 4 ≫ app₄.hom = app₃.hom ≫ g.map' 3 4)
                                             /-
                                               🎉 no goals
                                             -/
          /-
            C : Type u_1
            inst✝ : CategoryTheory.Category.{?u.412135, u_1} C
            n m : Nat
            F G : CategoryTheory.ComposableArrows C n
            f g : CategoryTheory.ComposableArrows C 5
            app₀ : CategoryTheory.Iso (f.obj' 0 ⋯) (g.obj' 0 ⋯)
            app₁ : CategoryTheory.Iso (f.obj' 1 ⋯) (g.obj' 1 ⋯)
            app₂ : CategoryTheory.Iso (f.obj' 2 ⋯) (g.obj' 2 ⋯)
            app₃ : CategoryTheory.Iso (f.obj' 3 ⋯) (g.obj' 3 ⋯)
            app₄ : CategoryTheory.Iso (f.obj' 4 ⋯) (g.obj' 4 ⋯)
            app₅ : CategoryTheory.Iso (f.obj' 5 ⋯) (g.obj' 5 ⋯)
            w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁.hom) (Catego …
            w₁ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 1 2 ⋯ ⋯) app₂.hom) (Catego …
            w₂ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 2 3 ⋯ ⋯) app₃.hom) (Catego …
            w₃ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 3 4 ⋯ ⋯) app₄.hom) (Catego …
            ⊢ LE.le 4 5
          -/
          /-
            🎉 no goals
          -/
          /-
            🎉 no goals
          -/
                                             /-
                                               🎉 no goals
                                             -/
    (w₄ : f.map' 4 5 ≫ app₅.hom = app₄.hom ≫ g.map' 4 5) :
                                             /-
                                               🎉 no goals
                                             -/
    f ≅ g where
  hom := homMk₅ app₀.hom app₁.hom app₂.hom app₃.hom app₄.hom app₅.hom w₀ w₁ w₂ w₃ w₄
  inv := homMk₅ app₀.inv app₁.inv app₂.inv app₃.inv app₄.inv app₅.inv
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.412135, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 5
          app₀ : CategoryTheory.Iso (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : CategoryTheory.Iso (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : CategoryTheory.Iso (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : CategoryTheory.Iso (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          app₄ : CategoryTheory.Iso (f.obj' 4 ⋯) (g.obj' 4 ⋯)
          app₅ : CategoryTheory.Iso (f.obj' 5 ⋯) (g.obj' 5 ⋯)
          w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁.hom) (Catego …
          w₁ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 1 2 ⋯ ⋯) app₂.hom) (Catego …
          w₂ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 2 3 ⋯ ⋯) app₃.hom) (Catego …
          w₃ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 3 4 ⋯ ⋯) app₄.hom) (Catego …
          w₄ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 4 5 ⋯ ⋯) app₅.hom) (Catego …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (g.map' 0 1 ⋯ ⋯) app₁.inv) (CategoryT …
        -/
    (by rw [map'_inv_eq_inv_map' (by valid) app₀ app₁ w₀])
        /-
          🎉 no goals
        -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.412135, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 5
          app₀ : CategoryTheory.Iso (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : CategoryTheory.Iso (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : CategoryTheory.Iso (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : CategoryTheory.Iso (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          app₄ : CategoryTheory.Iso (f.obj' 4 ⋯) (g.obj' 4 ⋯)
          app₅ : CategoryTheory.Iso (f.obj' 5 ⋯) (g.obj' 5 ⋯)
          w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁.hom) (Catego …
          w₁ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 1 2 ⋯ ⋯) app₂.hom) (Catego …
          w₂ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 2 3 ⋯ ⋯) app₃.hom) (Catego …
          w₃ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 3 4 ⋯ ⋯) app₄.hom) (Catego …
          w₄ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 4 5 ⋯ ⋯) app₅.hom) (Catego …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (g.map' 1 2 ⋯ ⋯) app₂.inv) (CategoryT …
        -/
    (by rw [map'_inv_eq_inv_map' (by valid) app₁ app₂ w₁])
        /-
          🎉 no goals
        -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.412135, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 5
          app₀ : CategoryTheory.Iso (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : CategoryTheory.Iso (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : CategoryTheory.Iso (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : CategoryTheory.Iso (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          app₄ : CategoryTheory.Iso (f.obj' 4 ⋯) (g.obj' 4 ⋯)
          app₅ : CategoryTheory.Iso (f.obj' 5 ⋯) (g.obj' 5 ⋯)
          w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁.hom) (Catego …
          w₁ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 1 2 ⋯ ⋯) app₂.hom) (Catego …
          w₂ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 2 3 ⋯ ⋯) app₃.hom) (Catego …
          w₃ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 3 4 ⋯ ⋯) app₄.hom) (Catego …
          w₄ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 4 5 ⋯ ⋯) app₅.hom) (Catego …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (g.map' 2 3 ⋯ ⋯) app₃.inv) (CategoryT …
        -/
    (by rw [map'_inv_eq_inv_map' (by valid) app₂ app₃ w₂])
        /-
          🎉 no goals
        -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.412135, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 5
          app₀ : CategoryTheory.Iso (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : CategoryTheory.Iso (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : CategoryTheory.Iso (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : CategoryTheory.Iso (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          app₄ : CategoryTheory.Iso (f.obj' 4 ⋯) (g.obj' 4 ⋯)
          app₅ : CategoryTheory.Iso (f.obj' 5 ⋯) (g.obj' 5 ⋯)
          w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁.hom) (Catego …
          w₁ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 1 2 ⋯ ⋯) app₂.hom) (Catego …
          w₂ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 2 3 ⋯ ⋯) app₃.hom) (Catego …
          w₃ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 3 4 ⋯ ⋯) app₄.hom) (Catego …
          w₄ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 4 5 ⋯ ⋯) app₅.hom) (Catego …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (g.map' 3 4 ⋯ ⋯) app₄.inv) (CategoryT …
        -/
    (by rw [map'_inv_eq_inv_map' (by valid) app₃ app₄ w₃])
        /-
          🎉 no goals
        -/
        /-
          C : Type u_1
          inst✝ : CategoryTheory.Category.{?u.412135, u_1} C
          n m : Nat
          F G : CategoryTheory.ComposableArrows C n
          f g : CategoryTheory.ComposableArrows C 5
          app₀ : CategoryTheory.Iso (f.obj' 0 ⋯) (g.obj' 0 ⋯)
          app₁ : CategoryTheory.Iso (f.obj' 1 ⋯) (g.obj' 1 ⋯)
          app₂ : CategoryTheory.Iso (f.obj' 2 ⋯) (g.obj' 2 ⋯)
          app₃ : CategoryTheory.Iso (f.obj' 3 ⋯) (g.obj' 3 ⋯)
          app₄ : CategoryTheory.Iso (f.obj' 4 ⋯) (g.obj' 4 ⋯)
          app₅ : CategoryTheory.Iso (f.obj' 5 ⋯) (g.obj' 5 ⋯)
          w₀ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 0 1 ⋯ ⋯) app₁.hom) (Catego …
          w₁ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 1 2 ⋯ ⋯) app₂.hom) (Catego …
          w₂ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 2 3 ⋯ ⋯) app₃.hom) (Catego …
          w₃ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 3 4 ⋯ ⋯) app₄.hom) (Catego …
          w₄ : Eq (CategoryTheory.CategoryStruct.comp (f.map' 4 5 ⋯ ⋯) app₅.hom) (Catego …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (g.map' 4 5 ⋯ ⋯) app₅.inv) (CategoryT …
        -/
    (by rw [map'_inv_eq_inv_map' (by valid) app₄ app₅ w₄])
        /-
          🎉 no goals
        -/


lemma ext₅ {f g : ComposableArrows C 5}
          /-
            C : Type u_1
            inst✝ : CategoryTheory.Category.{?u.442605, u_1} C
            n m : Nat
            F G : CategoryTheory.ComposableArrows C n
            f g : CategoryTheory.ComposableArrows C 5
            ⊢ LE.le 0 5
          -/
          /-
            🎉 no goals
          -/
                     /-
                       🎉 no goals
                     -/
                                     /-
                                       🎉 no goals
                                     -/
                                                /-
                                                  🎉 no goals
                                                -/
                                                                /-
                                                                  🎉 no goals
                                                                -/
    (h₀ : f.obj' 0 = g.obj' 0) (h₁ : f.obj' 1 = g.obj' 1) (h₂ : f.obj' 2 = g.obj' 2)
                                                                           /-
                                                                             🎉 no goals
                                                                           -/
          /-
            C : Type u_1
            inst✝ : CategoryTheory.Category.{?u.442605, u_1} C
            n m : Nat
            F G : CategoryTheory.ComposableArrows C n
            f g : CategoryTheory.ComposableArrows C 5
            h₀ : Eq (f.obj' 0 ⋯) (g.obj' 0 ⋯)
            h₁ : Eq (f.obj' 1 ⋯) (g.obj' 1 ⋯)
            h₂ : Eq (f.obj' 2 ⋯) (g.obj' 2 ⋯)
            ⊢ LE.le 3 5
          -/
          /-
            🎉 no goals
          -/
                     /-
                       🎉 no goals
                     -/
                                     /-
                                       🎉 no goals
                                     -/
                                                /-
                                                  🎉 no goals
                                                -/
                                                                /-
                                                                  🎉 no goals
                                                                -/
    (h₃ : f.obj' 3 = g.obj' 3) (h₄ : f.obj' 4 = g.obj' 4) (h₅ : f.obj' 5 = g.obj' 5)
                                                                           /-
                                                                             🎉 no goals
                                                                           -/
          /-
            C : Type u_1
            inst✝ : CategoryTheory.Category.{?u.442605, u_1} C
            n m : Nat
            F G : CategoryTheory.ComposableArrows C n
            f g : CategoryTheory.ComposableArrows C 5
            h₀ : Eq (f.obj' 0 ⋯) (g.obj' 0 ⋯)
            h₁ : Eq (f.obj' 1 ⋯) (g.obj' 1 ⋯)
            h₂ : Eq (f.obj' 2 ⋯) (g.obj' 2 ⋯)
            h₃ : Eq (f.obj' 3 ⋯) (g.obj' 3 ⋯)
            h₄ : Eq (f.obj' 4 ⋯) (g.obj' 4 ⋯)
            h₅ : Eq (f.obj' 5 ⋯) (g.obj' 5 ⋯)
            ⊢ LE.le 0 1
          -/
          /-
            🎉 no goals
          -/
          /-
            🎉 no goals
          -/
                                    /-
                                      🎉 no goals
                                    -/
    (w₀ : f.map' 0 1 = eqToHom h₀ ≫ g.map' 0 1 ≫ eqToHom h₁.symm)
                                    /-
                                      🎉 no goals
                                    -/
          /-
            C : Type u_1
            inst✝ : CategoryTheory.Category.{?u.442605, u_1} C
            n m : Nat
            F G : CategoryTheory.ComposableArrows C n
            f g : CategoryTheory.ComposableArrows C 5
            h₀ : Eq (f.obj' 0 ⋯) (g.obj' 0 ⋯)
            h₁ : Eq (f.obj' 1 ⋯) (g.obj' 1 ⋯)
            h₂ : Eq (f.obj' 2 ⋯) (g.obj' 2 ⋯)
            h₃ : Eq (f.obj' 3 ⋯) (g.obj' 3 ⋯)
            h₄ : Eq (f.obj' 4 ⋯) (g.obj' 4 ⋯)
            h₅ : Eq (f.obj' 5 ⋯) (g.obj' 5 ⋯)
            w₀ : Eq (f.map' 0 1 ⋯ ⋯) (CategoryTheory.CategoryStruct.comp (CategoryTheory.e …
            ⊢ LE.le 1 2
          -/
          /-
            🎉 no goals
          -/
          /-
            🎉 no goals
          -/
                                    /-
                                      🎉 no goals
                                    -/
    (w₁ : f.map' 1 2 = eqToHom h₁ ≫ g.map' 1 2 ≫ eqToHom h₂.symm)
                                    /-
                                      🎉 no goals
                                    -/
          /-
            C : Type u_1
            inst✝ : CategoryTheory.Category.{?u.442605, u_1} C
            n m : Nat
            F G : CategoryTheory.ComposableArrows C n
            f g : CategoryTheory.ComposableArrows C 5
            h₀ : Eq (f.obj' 0 ⋯) (g.obj' 0 ⋯)
            h₁ : Eq (f.obj' 1 ⋯) (g.obj' 1 ⋯)
            h₂ : Eq (f.obj' 2 ⋯) (g.obj' 2 ⋯)
            h₃ : Eq (f.obj' 3 ⋯) (g.obj' 3 ⋯)
            h₄ : Eq (f.obj' 4 ⋯) (g.obj' 4 ⋯)
            h₅ : Eq (f.obj' 5 ⋯) (g.obj' 5 ⋯)
            w₀ : Eq (f.map' 0 1 ⋯ ⋯) (CategoryTheory.CategoryStruct.comp (CategoryTheory.e …
            w₁ : Eq (f.map' 1 2 ⋯ ⋯) (CategoryTheory.CategoryStruct.comp (CategoryTheory.e …
            ⊢ LE.le 2 3
          -/
          /-
            🎉 no goals
          -/
          /-
            🎉 no goals
          -/
                                    /-
                                      🎉 no goals
                                    -/
    (w₂ : f.map' 2 3 = eqToHom h₂ ≫ g.map' 2 3 ≫ eqToHom h₃.symm)
                                    /-
                                      🎉 no goals
                                    -/
          /-
            C : Type u_1
            inst✝ : CategoryTheory.Category.{?u.442605, u_1} C
            n m : Nat
            F G : CategoryTheory.ComposableArrows C n
            f g : CategoryTheory.ComposableArrows C 5
            h₀ : Eq (f.obj' 0 ⋯) (g.obj' 0 ⋯)
            h₁ : Eq (f.obj' 1 ⋯) (g.obj' 1 ⋯)
            h₂ : Eq (f.obj' 2 ⋯) (g.obj' 2 ⋯)
            h₃ : Eq (f.obj' 3 ⋯) (g.obj' 3 ⋯)
            h₄ : Eq (f.obj' 4 ⋯) (g.obj' 4 ⋯)
            h₅ : Eq (f.obj' 5 ⋯) (g.obj' 5 ⋯)
            w₀ : Eq (f.map' 0 1 ⋯ ⋯) (CategoryTheory.CategoryStruct.comp (CategoryTheory.e …
            w₁ : Eq (f.map' 1 2 ⋯ ⋯) (CategoryTheory.CategoryStruct.comp (CategoryTheory.e …
            w₂ : Eq (f.map' 2 3 ⋯ ⋯) (CategoryTheory.CategoryStruct.comp (CategoryTheory.e …
            ⊢ LE.le 3 4
          -/
          /-
            🎉 no goals
          -/
          /-
            🎉 no goals
          -/
                                    /-
                                      🎉 no goals
                                    -/
    (w₃ : f.map' 3 4 = eqToHom h₃ ≫ g.map' 3 4 ≫ eqToHom h₄.symm)
                                    /-
                                      🎉 no goals
                                    -/
          /-
            C : Type u_1
            inst✝ : CategoryTheory.Category.{?u.442605, u_1} C
            n m : Nat
            F G : CategoryTheory.ComposableArrows C n
            f g : CategoryTheory.ComposableArrows C 5
            h₀ : Eq (f.obj' 0 ⋯) (g.obj' 0 ⋯)
            h₁ : Eq (f.obj' 1 ⋯) (g.obj' 1 ⋯)
            h₂ : Eq (f.obj' 2 ⋯) (g.obj' 2 ⋯)
            h₃ : Eq (f.obj' 3 ⋯) (g.obj' 3 ⋯)
            h₄ : Eq (f.obj' 4 ⋯) (g.obj' 4 ⋯)
            h₅ : Eq (f.obj' 5 ⋯) (g.obj' 5 ⋯)
            w₀ : Eq (f.map' 0 1 ⋯ ⋯) (CategoryTheory.CategoryStruct.comp (CategoryTheory.e …
            w₁ : Eq (f.map' 1 2 ⋯ ⋯) (CategoryTheory.CategoryStruct.comp (CategoryTheory.e …
            w₂ : Eq (f.map' 2 3 ⋯ ⋯) (CategoryTheory.CategoryStruct.comp (CategoryTheory.e …
            w₃ : Eq (f.map' 3 4 ⋯ ⋯) (CategoryTheory.CategoryStruct.comp (CategoryTheory.e …
            ⊢ LE.le 4 5
          -/
          /-
            🎉 no goals
          -/
          /-
            🎉 no goals
          -/
                                    /-
                                      🎉 no goals
                                    -/
    (w₄ : f.map' 4 5 = eqToHom h₄ ≫ g.map' 4 5 ≫ eqToHom h₅.symm) :
                                    /-
                                      🎉 no goals
                                    -/
    f = g :=
  ext_succ h₀ (ext₄ h₁ h₂ h₃ h₄ h₅ w₁ w₂ w₃ w₄) w₀


lemma mk₅_surjective (X : ComposableArrows C 5) :
    ∃ (X₀ X₁ X₂ X₃ X₄ X₅ : C) (f₀ : X₀ ⟶ X₁) (f₁ : X₁ ⟶ X₂) (f₂ : X₂ ⟶ X₃)
      (f₃ : X₃ ⟶ X₄) (f₄ : X₄ ⟶ X₅), X = mk₅ f₀ f₁ f₂ f₃ f₄ :=
                     /-
                       C : Type u_1
                       inst✝ : CategoryTheory.Category.{u_2, u_1} C
                       X : CategoryTheory.ComposableArrows C 5
                       ⊢ LE.le 0 1
                     -/
                     /-
                       🎉 no goals
                     -/
                     /-
                       🎉 no goals
                     -/
                                 /-
                                   🎉 no goals
                                 -/
                                 /-
                                   🎉 no goals
                                 -/
                                             /-
                                               🎉 no goals
                                             -/
                                             /-
                                               🎉 no goals
                                             -/
                                                         /-
                                                           🎉 no goals
                                                         -/
                                                         /-
                                                           🎉 no goals
                                                         -/
                                                                     /-
                                                                       🎉 no goals
                                                                     -/
  ⟨_, _, _, _, _, _, X.map' 0 1, X.map' 1 2, X.map' 2 3, X.map' 3 4, X.map' 4 5,
                                                                     /-
                                                                       🎉 no goals
                                                                     -/
                                     /-
                                       C : Type u_1
                                       inst✝ : CategoryTheory.Category.{u_2, u_1} C
                                       X : CategoryTheory.ComposableArrows C 5
                                       ⊢ Eq (X.map' 0 1 ⋯ ⋯) (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqTo …
                                     -/
                                     /-
                                       🎉 no goals
                                     -/
                                               /-
                                                 🎉 no goals
                                               -/
                                                         /-
                                                           🎉 no goals
                                                         -/
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
    ext₅ rfl rfl rfl rfl rfl rfl (by simp) (by simp) (by simp) (by simp) (by simp)⟩
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


/-- The `i`th arrow of `F : ComposableArrows C n`. -/
def arrow (i : ℕ) (hi : i < n := by valid) :
                                 /-
                                   C : Type u_1
                                   inst✝ : CategoryTheory.Category.{?u.495714, u_1} C
                                   n m : Nat
                                   F G : CategoryTheory.ComposableArrows C n
                                   i : Nat
                                   hi : autoParam (LT.lt i n) _auto✝
                                   ⊢ LE.le i (HAdd.hAdd i 1)
                                 -/
                                 /-
                                   🎉 no goals
                                 -/
    ComposableArrows C 1 := mk₁ (F.map' i (i + 1))
                                 /-
                                   🎉 no goals
                                 -/


lemma mkOfObjOfMapSucc_exists : ∃ (F : ComposableArrows C n) (e : ∀ i, F.obj i ≅ obj i),
    ∀ (i : ℕ) (hi : i < n), mapSucc ⟨i, hi⟩ =
                       /-
                         C : Type u_1
                         inst✝ : CategoryTheory.Category.{?u.496289, u_1} C
                         n m : Nat
                         F✝ G : CategoryTheory.ComposableArrows C n
                         obj : Fin (HAdd.hAdd n 1) → C
                         mapSucc : (i : Fin n) → Quiver.Hom (obj i.castSucc) (obj i.succ)
                         F : CategoryTheory.ComposableArrows C n
                         e : (i : Fin (HAdd.hAdd n 1)) → CategoryTheory.Iso (F.obj i) (obj i)
                         i : Nat
                         hi : LT.lt i n
                         ⊢ LE.le i (HAdd.hAdd i 1)
                       -/
                       /-
                         🎉 no goals
                       -/
      (e ⟨i, _⟩).inv ≫ F.map' i (i + 1) ≫ (e ⟨i + 1, _⟩).hom := by
                       /-
                         🎉 no goals
                       -/
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    n : Nat
    obj : Fin (HAdd.hAdd n 1) → C
    mapSucc : (i : Fin n) → Quiver.Hom (obj i.castSucc) (obj i.succ)
    ⊢ Exists fun F => Exists fun e => ∀ (i : Nat) (hi : LT.lt i n), Eq (mapSucc ⟨i …
  -/
  revert obj mapSucc
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    n : Nat
    ⊢ ∀ (obj : Fin (HAdd.hAdd n 1) → C) (mapSucc : (i : Fin n) → Quiver.Hom (obj i …
  -/
  induction' n with n hn
    /-
      case zero
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      ⊢ ∀ (obj : Fin (HAdd.hAdd 0 1) → C) (mapSucc : (i : Fin 0) → Quiver.Hom (obj i …
    -/
  · intro obj _
    /-
      case zero
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      obj : Fin (HAdd.hAdd 0 1) → C
      mapSucc✝ : (i : Fin 0) → Quiver.Hom (obj i.castSucc) (obj i.succ)
      ⊢ Exists fun F => Exists fun e => ∀ (i : Nat) (hi : LT.lt i 0), Eq (mapSucc✝ ⟨ …
    -/
    exact ⟨mk₀ (obj 0), fun 0 => Iso.refl _, fun i hi => by simp at hi⟩
    /-
      🎉 no goals
    -/
    /-
      case succ
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      n : Nat
      hn : ∀ (obj : Fin (HAdd.hAdd n 1) → C) (mapSucc : (i : Fin n) → Quiver.Hom (ob …
      ⊢ ∀ (obj : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1) → C) (mapSucc : (i : Fin (HAdd.hA …
    -/
  · intro obj mapSucc
    /-
      case succ
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      n : Nat
      hn : ∀ (obj : Fin (HAdd.hAdd n 1) → C) (mapSucc : (i : Fin n) → Quiver.Hom (ob …
      obj : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1) → C
      mapSucc : (i : Fin (HAdd.hAdd n 1)) → Quiver.Hom (obj i.castSucc) (obj i.succ)
      ⊢ Exists fun F => Exists fun e => ∀ (i : Nat) (hi : LT.lt i (HAdd.hAdd n 1)),  …
    -/
    obtain ⟨F, e, h⟩ := hn (fun i => obj i.succ) (fun i => mapSucc i.succ)
    refine ⟨F.precomp (mapSucc 0 ≫ (e 0).inv), fun i => match i with
      | 0 => Iso.refl _
      | ⟨i + 1, hi⟩ => e _, fun i hi => ?_⟩
    /-
      case succ.intro.intro
      C : Type u_1
      inst✝ : CategoryTheory.Category.{u_2, u_1} C
      n : Nat
      hn : ∀ (obj : Fin (HAdd.hAdd n 1) → C) (mapSucc : (i : Fin n) → Quiver.Hom (ob …
      obj : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1) → C
      mapSucc : (i : Fin (HAdd.hAdd n 1)) → Quiver.Hom (obj i.castSucc) (obj i.succ)
      F : CategoryTheory.ComposableArrows C n
      e : (i : Fin (HAdd.hAdd n 1)) → CategoryTheory.Iso (F.obj i) (obj i.succ)
      h : ∀ (i : Nat) (hi : LT.lt i n), Eq (mapSucc ⟨i, hi⟩.succ) (CategoryTheory.Ca …
      i : Nat
      hi : LT.lt i (HAdd.hAdd n 1)
      ⊢ Eq (mapSucc ⟨i, hi⟩) (CategoryTheory.CategoryStruct.comp ((fun i => Category …
    -/
    obtain _ | i := i
      /-
        case succ.intro.intro.zero
        C : Type u_1
        inst✝ : CategoryTheory.Category.{u_2, u_1} C
        n : Nat
        hn : ∀ (obj : Fin (HAdd.hAdd n 1) → C) (mapSucc : (i : Fin n) → Quiver.Hom (ob …
        obj : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1) → C
        mapSucc : (i : Fin (HAdd.hAdd n 1)) → Quiver.Hom (obj i.castSucc) (obj i.succ)
        F : CategoryTheory.ComposableArrows C n
        e : (i : Fin (HAdd.hAdd n 1)) → CategoryTheory.Iso (F.obj i) (obj i.succ)
        h : ∀ (i : Nat) (hi : LT.lt i n), Eq (mapSucc ⟨i, hi⟩.succ) (CategoryTheory.Ca …
        hi : LT.lt 0 (HAdd.hAdd n 1)
        ⊢ Eq (mapSucc ⟨0, hi⟩) (CategoryTheory.CategoryStruct.comp ((fun i => Category …
      -/
    · dsimp
      /-
        case succ.intro.intro.zero
        C : Type u_1
        inst✝ : CategoryTheory.Category.{u_2, u_1} C
        n : Nat
        hn : ∀ (obj : Fin (HAdd.hAdd n 1) → C) (mapSucc : (i : Fin n) → Quiver.Hom (ob …
        obj : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1) → C
        mapSucc : (i : Fin (HAdd.hAdd n 1)) → Quiver.Hom (obj i.castSucc) (obj i.succ)
        F : CategoryTheory.ComposableArrows C n
        e : (i : Fin (HAdd.hAdd n 1)) → CategoryTheory.Iso (F.obj i) (obj i.succ)
        h : ∀ (i : Nat) (hi : LT.lt i n), Eq (mapSucc ⟨i, hi⟩.succ) (CategoryTheory.Ca …
        hi : LT.lt 0 (HAdd.hAdd n 1)
        ⊢ Eq (mapSucc 0) (CategoryTheory.CategoryStruct.comp (CategoryTheory.Composabl …
      -/
      rw [assoc, Iso.inv_hom_id, comp_id]
      /-
        case succ.intro.intro.zero
        C : Type u_1
        inst✝ : CategoryTheory.Category.{u_2, u_1} C
        n : Nat
        hn : ∀ (obj : Fin (HAdd.hAdd n 1) → C) (mapSucc : (i : Fin n) → Quiver.Hom (ob …
        obj : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1) → C
        mapSucc : (i : Fin (HAdd.hAdd n 1)) → Quiver.Hom (obj i.castSucc) (obj i.succ)
        F : CategoryTheory.ComposableArrows C n
        e : (i : Fin (HAdd.hAdd n 1)) → CategoryTheory.Iso (F.obj i) (obj i.succ)
        h : ∀ (i : Nat) (hi : LT.lt i n), Eq (mapSucc ⟨i, hi⟩.succ) (CategoryTheory.Ca …
        hi : LT.lt 0 (HAdd.hAdd n 1)
        ⊢ Eq (mapSucc 0) (CategoryTheory.CategoryStruct.comp (CategoryTheory.Composabl …
      -/
      erw [id_comp]
      /-
        🎉 no goals
      -/
      /-
        case succ.intro.intro.succ
        C : Type u_1
        inst✝ : CategoryTheory.Category.{u_2, u_1} C
        n : Nat
        hn : ∀ (obj : Fin (HAdd.hAdd n 1) → C) (mapSucc : (i : Fin n) → Quiver.Hom (ob …
        obj : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1) → C
        mapSucc : (i : Fin (HAdd.hAdd n 1)) → Quiver.Hom (obj i.castSucc) (obj i.succ)
        F : CategoryTheory.ComposableArrows C n
        e : (i : Fin (HAdd.hAdd n 1)) → CategoryTheory.Iso (F.obj i) (obj i.succ)
        h : ∀ (i : Nat) (hi : LT.lt i n), Eq (mapSucc ⟨i, hi⟩.succ) (CategoryTheory.Ca …
        i : Nat
        hi : LT.lt (HAdd.hAdd i 1) (HAdd.hAdd n 1)
        ⊢ Eq (mapSucc ⟨HAdd.hAdd i 1, hi⟩) (CategoryTheory.CategoryStruct.comp ((fun i …
      -/
    · exact h i (by valid)
      /-
        🎉 no goals
      -/


/-- Given `obj : Fin (n + 1) → C` and `mapSucc i : obj i.castSucc ⟶ obj i.succ`
for all `i : Fin n`, this is `F : ComposableArrows C n` such that `F.obj i` is
definitionally equal to `obj i` and such that `F.map' i (i + 1) = mapSucc ⟨i, hi⟩`. -/
noncomputable def mkOfObjOfMapSucc : ComposableArrows C n :=
  (mkOfObjOfMapSucc_exists obj mapSucc).choose.copyObj obj
    (mkOfObjOfMapSucc_exists obj mapSucc).choose_spec.choose


@[simp]
lemma mkOfObjOfMapSucc_obj (i : Fin (n + 1)) :
    (mkOfObjOfMapSucc obj mapSucc).obj i = obj i := rfl


lemma mkOfObjOfMapSucc_map_succ (i : ℕ) (hi : i < n := by valid) :
    /-
      C : Type u_1
      inst✝ : CategoryTheory.Category.{?u.505248, u_1} C
      n m : Nat
      F G : CategoryTheory.ComposableArrows C n
      obj : Fin (HAdd.hAdd n 1) → C
      mapSucc : (i : Fin n) → Quiver.Hom (obj i.castSucc) (obj i.succ)
      i : Nat
      hi : autoParam (LT.lt i n) _auto✝
      ⊢ LE.le i (HAdd.hAdd i 1)
    -/
    /-
      🎉 no goals
    -/
    (mkOfObjOfMapSucc obj mapSucc).map' i (i + 1) = mapSucc ⟨i, hi⟩ :=
    /-
      🎉 no goals
    -/
  ((mkOfObjOfMapSucc_exists obj mapSucc).choose_spec.choose_spec i hi).symm


lemma mkOfObjOfMapSucc_arrow (i : ℕ) (hi : i < n := by valid) :
    /-
      C : Type u_1
      inst✝ : CategoryTheory.Category.{?u.505890, u_1} C
      n m : Nat
      F G : CategoryTheory.ComposableArrows C n
      obj : Fin (HAdd.hAdd n 1) → C
      mapSucc : (i : Fin n) → Quiver.Hom (obj i.castSucc) (obj i.succ)
      i : Nat
      hi : autoParam (LT.lt i n) _auto✝
      ⊢ LT.lt i n
    -/
    (mkOfObjOfMapSucc obj mapSucc).arrow i = mk₁ (mapSucc ⟨i, hi⟩) :=
    /-
      🎉 no goals
    -/
                   /-
                     C : Type u_1
                     inst✝ : CategoryTheory.Category.{u_2, u_1} C
                     n : Nat
                     obj : Fin (HAdd.hAdd n 1) → C
                     mapSucc : (i : Fin n) → Quiver.Hom (obj i.castSucc) (obj i.succ)
                     i : Nat
                     hi : autoParam (LT.lt i n) _auto✝
                     ⊢ Eq ((CategoryTheory.ComposableArrows.mkOfObjOfMapSucc obj mapSucc).arrow i h …
                   -/
  ext₁ rfl rfl (by simpa using mkOfObjOfMapSucc_map_succ obj mapSucc i hi)
                   /-
                     🎉 no goals
                   -/


suppress_compilation in
variable (C n) in
/-- The equivalence `(ComposableArrows C n)ᵒᵖ ≌ ComposableArrows Cᵒᵖ n` obtained
by reversing the arrows. -/
@[simps!]
def opEquivalence : (ComposableArrows C n)ᵒᵖ ≌ ComposableArrows Cᵒᵖ n :=
  ((orderDualEquivalence (Fin (n + 1))).symm.trans
      Fin.revOrderIso.equivalence).symm.congrLeft.op.trans
    (Functor.leftOpRightOpEquiv (Fin (n + 1)) C)


/-- The functor `ComposableArrows C n ⥤ ComposableArrows D n` obtained by postcomposition
with a functor `C ⥤ D`. -/
@[simps!]
def Functor.mapComposableArrows :
    ComposableArrows C n ⥤ ComposableArrows D n :=
  (whiskeringRight _ _ _).obj G


suppress_compilation in
/-- The functor `ComposableArrows C n ⥤ ComposableArrows D n` induced by `G : C ⥤ D`
commutes with `opEquivalence`. -/
def Functor.mapComposableArrowsOpIso :
    G.mapComposableArrows n ⋙ (opEquivalence D n).functor.rightOp ≅
      (opEquivalence C n).functor.rightOp ⋙ (G.op.mapComposableArrows n).op :=
  Iso.refl _


