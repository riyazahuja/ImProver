lemma congr_f (i j : ℕ) (h : i = j) :
                      /-
                        C : Type u_1
                        inst✝ : CategoryTheory.Category.{?u.125, u_1} C
                        X : Nat → C
                        f : (n : Nat) → Quiver.Hom (X n) (X (HAdd.hAdd n 1))
                        i j : Nat
                        h : Eq i j
                        ⊢ Eq (X i) (X j)
                      -/
                      /-
                        🎉 no goals
                      -/
    f i = eqToHom (by rw [h]) ≫ f j ≫ eqToHom (by rw [h]) := by
                                                  /-
                                                    🎉 no goals
                                                  -/
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    X : Nat → C
    f : (n : Nat) → Quiver.Hom (X n) (X (HAdd.hAdd n 1))
    i j : Nat
    h : Eq i j
    ⊢ Eq (f i) (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (Cat …
  -/
  subst h
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    X : Nat → C
    f : (n : Nat) → Quiver.Hom (X n) (X (HAdd.hAdd n 1))
    i : Nat
    ⊢ Eq (f i) (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (Cat …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The morphism `X i ⟶ X j` obtained by composing morphisms of
the form `X n ⟶ X (n + 1)` when `i ≤ j`. -/
def map : ∀ {X : ℕ → C} (_ : ∀ n, X n ⟶ X (n + 1)) (i j : ℕ), i ≤ j → (X i ⟶ X j)
  | _, _, 0, 0 => fun _ ↦ 𝟙 _
  | _, f, 0, 1 => fun _ ↦ f 0
                                                                    /-
                                                                      C : Type u_1
                                                                      inst✝ : CategoryTheory.Category.{?u.787, u_1} C
                                                                      X : Nat → C
                                                                      f✝ : (n : Nat) → Quiver.Hom (X n) (X (HAdd.hAdd n 1))
                                                                      x✝¹ : Nat → C
                                                                      f : (n : Nat) → Quiver.Hom (x✝¹ n) (x✝¹ (HAdd.hAdd n 1))
                                                                      l : Nat
                                                                      x✝ : LE.le 0 (HAdd.hAdd l 1)
                                                                      ⊢ LE.le 0 l
                                                                    -/
  | _, f, 0, l + 1 => fun _ ↦ f 0 ≫ map (fun n ↦ f (n + 1)) 0 l (by omega)
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
  | _, _, _ + 1, 0 => nofun
                                                                  /-
                                                                    C : Type u_1
                                                                    inst✝ : CategoryTheory.Category.{?u.787, u_1} C
                                                                    X : Nat → C
                                                                    f✝ : (n : Nat) → Quiver.Hom (X n) (X (HAdd.hAdd n 1))
                                                                    x✝¹ : Nat → C
                                                                    f : (n : Nat) → Quiver.Hom (x✝¹ n) (x✝¹ (HAdd.hAdd n 1))
                                                                    k l : Nat
                                                                    x✝ : LE.le (HAdd.hAdd k 1) (HAdd.hAdd l 1)
                                                                    ⊢ LE.le k l
                                                                  -/
  | _, f, k + 1, l + 1 => fun _ ↦ map (fun n ↦ f (n + 1)) k l (by omega)
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


                                     /-
                                       C : Type u_1
                                       inst✝ : CategoryTheory.Category.{?u.2994, u_1} C
                                       X : Nat → C
                                       f : (n : Nat) → Quiver.Hom (X n) (X (HAdd.hAdd n 1))
                                       i : Nat
                                       ⊢ LE.le i i
                                     -/
lemma map_id (i : ℕ) : map f i i (by omega) = 𝟙 _ := by
                                     /-
                                       🎉 no goals
                                     -/
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    X : Nat → C
    f : (n : Nat) → Quiver.Hom (X n) (X (HAdd.hAdd n 1))
    i : Nat
    ⊢ Eq (CategoryTheory.Functor.OfSequence.map f i i ⋯) (CategoryTheory.CategoryS …
  -/
  revert X f
  induction i with
  | zero => intros; rfl
  | succ _ hi =>
      intro X f
      apply hi


                                                /-
                                                  C : Type u_1
                                                  inst✝ : CategoryTheory.Category.{?u.3481, u_1} C
                                                  X : Nat → C
                                                  f : (n : Nat) → Quiver.Hom (X n) (X (HAdd.hAdd n 1))
                                                  i : Nat
                                                  ⊢ LE.le i (HAdd.hAdd i 1)
                                                -/
lemma map_le_succ (i : ℕ) : map f i (i + 1) (by omega) = f i := by
                                                /-
                                                  🎉 no goals
                                                -/
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    X : Nat → C
    f : (n : Nat) → Quiver.Hom (X n) (X (HAdd.hAdd n 1))
    i : Nat
    ⊢ Eq (CategoryTheory.Functor.OfSequence.map f i (HAdd.hAdd i 1) ⋯) (f i)
  -/
  revert X f
  induction i with
  | zero => intros; rfl
  | succ _ hi =>
      intro X f
      apply hi


@[reassoc]
lemma map_comp (i j k : ℕ) (hij : i ≤ j) (hjk : j ≤ k) :
    map f i k (hij.trans hjk) = map f i j hij ≫ map f j k hjk := by
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    X : Nat → C
    f : (n : Nat) → Quiver.Hom (X n) (X (HAdd.hAdd n 1))
    i j k : Nat
    hij : LE.le i j
    hjk : LE.le j k
    ⊢ Eq (CategoryTheory.Functor.OfSequence.map f i k ⋯) (CategoryTheory.CategoryS …
  -/
  revert X f j k
  induction i with
  | zero =>
      intros X f j
      revert X f
      induction j with
      | zero =>
          intros X f k hij hjk
          rw [map_id, id_comp]
      | succ j hj =>
          rintro X f (_|_|k) hij hjk
          · omega
          · obtain rfl : j = 0 := by omega
            rw [map_id, comp_id]
          · simp only [map, Nat.reduceAdd]
            rw [hj (fun n ↦ f (n + 1)) (k + 1) (by omega) (by omega)]
            obtain _|j := j
            all_goals simp [map]
  | succ i hi =>
      rintro X f (_|j) (_|k)
      · omega
      · omega
      · omega
      · intros
        exact hi _ j k (by omega) (by omega)

-- `map` has good definitional properties when applied to explicit natural numbers

/-- The functor `ℕ ⥤ C` constructed from a sequence of
morphisms `f : X n ⟶ X (n + 1)` for all `n : ℕ`. -/
@[simps obj]
def ofSequence : ℕ ⥤ C where
  obj := X
  map {i j} φ := OfSequence.map f i j (leOfHom φ)
  map_id i := OfSequence.map_id f i
  map_comp {i j k} α β := OfSequence.map_comp f i j k (leOfHom α) (leOfHom β)


@[simp]
lemma ofSequence_map_homOfLE_succ (n : ℕ) :
    (ofSequence f).map (homOfLE (Nat.le_add_right n 1)) = f n :=
  OfSequence.map_le_succ f n


/-- Constructor for natural transformations `F ⟶ G` in `ℕ ⥤ C` which takes as inputs
the morphisms `F.obj n ⟶ G.obj n` for all `n : ℕ` and the naturality condition only
for morphisms of the form `n ⟶ n + 1`. -/
@[simps app]
def ofSequence : F ⟶ G where
  app := app
  naturality := by
    /-
      C : Type u_1
      inst✝ : CategoryTheory.Category.{?u.16480, u_1} C
      F G : CategoryTheory.Functor Nat C
      app : (n : Nat) → Quiver.Hom (F.obj n) (G.obj n)
      naturality : ∀ (n : Nat), Eq (CategoryTheory.CategoryStruct.comp (F.map (Categ …
      ⊢ ∀ ⦃X Y : Nat⦄ (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp ( …
    -/
    intro i j φ
    /-
      C : Type u_1
      inst✝ : CategoryTheory.Category.{?u.16480, u_1} C
      F G : CategoryTheory.Functor Nat C
      app : (n : Nat) → Quiver.Hom (F.obj n) (G.obj n)
      naturality : ∀ (n : Nat), Eq (CategoryTheory.CategoryStruct.comp (F.map (Categ …
      i j : Nat
      φ : Quiver.Hom i j
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map φ) (app j)) (CategoryTheory.Ca …
    -/
    obtain ⟨k, hk⟩ := Nat.exists_eq_add_of_le (leOfHom φ)
    /-
      case intro
      C : Type u_1
      inst✝ : CategoryTheory.Category.{?u.16480, u_1} C
      F G : CategoryTheory.Functor Nat C
      app : (n : Nat) → Quiver.Hom (F.obj n) (G.obj n)
      naturality : ∀ (n : Nat), Eq (CategoryTheory.CategoryStruct.comp (F.map (Categ …
      i j : Nat
      φ : Quiver.Hom i j
      k : Nat
      hk : Eq j (HAdd.hAdd i k)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map φ) (app j)) (CategoryTheory.Ca …
    -/
    obtain rfl := Subsingleton.elim φ (homOfLE (by omega))
    /-
      case intro
      C : Type u_1
      inst✝ : CategoryTheory.Category.{?u.16480, u_1} C
      F G : CategoryTheory.Functor Nat C
      app : (n : Nat) → Quiver.Hom (F.obj n) (G.obj n)
      naturality : ∀ (n : Nat), Eq (CategoryTheory.CategoryStruct.comp (F.map (Categ …
      i j k : Nat
      hk : Eq j (HAdd.hAdd i k)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.homOfLE ⋯)) (a …
    -/
    revert i j
    induction k with
    | zero =>
        intro i j hk
        obtain rfl : j = i := by omega
        simp
    | succ k hk =>
        intro i j hk'
        obtain rfl : j = i + k + 1 := by omega
        simp only [← homOfLE_comp (show i ≤ i + k by omega) (show i + k ≤ i + k + 1 by omega),
          Functor.map_comp, assoc, naturality, reassoc_of% (hk rfl)]


/-- The functor `ℕᵒᵖ ⥤ C` constructed from a sequence of
morphisms `f : X (n + 1) ⟶ X n` for all `n : ℕ`. -/
@[simps! obj]
def ofOpSequence : ℕᵒᵖ ⥤ C := (ofSequence (fun n ↦ (f n).op)).leftOp

-- `ofOpSequence` has good definitional properties when applied to explicit natural numbers

@[simp]
lemma ofOpSequence_map_homOfLE_succ (n : ℕ) :
    (ofOpSequence f).map (homOfLE (Nat.le_add_right n 1)).op = f n := by
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    X : Nat → C
    f : (n : Nat) → Quiver.Hom (X (HAdd.hAdd n 1)) (X n)
    n : Nat
    ⊢ Eq ((CategoryTheory.Functor.ofOpSequence f).map (CategoryTheory.homOfLE ⋯).o …
  -/
  simp [ofOpSequence]
  /-
    🎉 no goals
  -/


/-- Constructor for natural transformations `F ⟶ G` in `ℕᵒᵖ ⥤ C` which takes as inputs
the morphisms `F.obj ⟨n⟩ ⟶ G.obj ⟨n⟩` for all `n : ℕ` and the naturality condition only
for morphisms of the form `n ⟶ n + 1`. -/
@[simps!]
def ofOpSequence : F ⟶ G where
  app n := app n.unop
  naturality _ _ f := by
    let φ : G.rightOp ⟶ F.rightOp := ofSequence (fun n ↦ (app n).op)
      (fun n ↦ Quiver.Hom.unop_inj (naturality n).symm)
    /-
      C : Type u_1
      inst✝ : CategoryTheory.Category.{?u.25390, u_1} C
      F G : CategoryTheory.Functor (Opposite Nat) C
      app : (n : Nat) → Quiver.Hom (F.obj { unop := n }) (G.obj { unop := n })
      naturality : ∀ (n : Nat), Eq (CategoryTheory.CategoryStruct.comp (F.map (Categ …
      x✝¹ x✝ : Opposite Nat
      f : Quiver.Hom x✝¹ x✝
      φ : Quiver.Hom G.rightOp F.rightOp := CategoryTheory.NatTrans.ofSequence (fun  …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map f) ((fun n => app (Opposite.un …
    -/
    exact Quiver.Hom.op_inj (φ.naturality f.unop).symm
    /-
      🎉 no goals
    -/


