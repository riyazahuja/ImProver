/-- The simplex category:
* objects are natural numbers `n : ℕ`
* morphisms from `n` to `m` are monotone functions `Fin (n+1) → Fin (m+1)`
-/
def SimplexCategory :=
  ℕ


/-- Interpret a natural number as an object of the simplex category. -/
def mk (n : ℕ) : SimplexCategory :=
  n


/-- the `n`-dimensional simplex can be denoted `[n]` -/
scoped[Simplicial] notation "[" n "]" => SimplexCategory.mk n

-- TODO: Make `len` irreducible.

/-- The length of an object of `SimplexCategory`. -/
def len (n : SimplexCategory) : ℕ :=
  n


@[ext]
theorem ext (a b : SimplexCategory) : a.len = b.len → a = b :=
  id


@[simp]
theorem len_mk (n : ℕ) : [n].len = n :=
  rfl


@[simp]
theorem mk_len (n : SimplexCategory) : ([n.len] : SimplexCategory) = n :=
  rfl


/-- A recursor for `SimplexCategory`. Use it as `induction Δ using SimplexCategory.rec`. -/
protected def rec {F : SimplexCategory → Sort*} (h : ∀ n : ℕ, F [n]) : ∀ X, F X := fun n =>
  h n.len

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): removed @[nolint has_nonempty_instance]

/-- Morphisms in the `SimplexCategory`. -/
protected def Hom (a b : SimplexCategory) :=
  Fin (a.len + 1) →o Fin (b.len + 1)


/-- Make a morphism in `SimplexCategory` from a monotone map of `Fin`'s. -/
def mk {a b : SimplexCategory} (f : Fin (a.len + 1) →o Fin (b.len + 1)) : SimplexCategory.Hom a b :=
  f


/-- Recover the monotone map from a morphism in the simplex category. -/
def toOrderHom {a b : SimplexCategory} (f : SimplexCategory.Hom a b) :
    Fin (a.len + 1) →o Fin (b.len + 1) :=
  f


theorem ext' {a b : SimplexCategory} (f g : SimplexCategory.Hom a b) :
    f.toOrderHom = g.toOrderHom → f = g :=
  id


@[simp]
theorem mk_toOrderHom {a b : SimplexCategory} (f : SimplexCategory.Hom a b) : mk f.toOrderHom = f :=
  rfl


@[simp]
theorem toOrderHom_mk {a b : SimplexCategory} (f : Fin (a.len + 1) →o Fin (b.len + 1)) :
    (mk f).toOrderHom = f :=
  rfl


theorem mk_toOrderHom_apply {a b : SimplexCategory} (f : Fin (a.len + 1) →o Fin (b.len + 1))
    (i : Fin (a.len + 1)) : (mk f).toOrderHom i = f i :=
  rfl


/-- Identity morphisms of `SimplexCategory`. -/
@[simp]
def id (a : SimplexCategory) : SimplexCategory.Hom a a :=
  mk OrderHom.id


/-- Composition of morphisms of `SimplexCategory`. -/
@[simp]
def comp {a b c : SimplexCategory} (f : SimplexCategory.Hom b c) (g : SimplexCategory.Hom a b) :
    SimplexCategory.Hom a c :=
  mk <| f.toOrderHom.comp g.toOrderHom


instance smallCategory : SmallCategory.{0} SimplexCategory where
  Hom n m := SimplexCategory.Hom n m
  id _ := SimplexCategory.Hom.id _
  comp f g := SimplexCategory.Hom.comp g f


@[simp]
lemma id_toOrderHom (a : SimplexCategory) :
    Hom.toOrderHom (𝟙 a) = OrderHom.id := rfl


@[simp]
lemma comp_toOrderHom {a b c : SimplexCategory} (f : a ⟶ b) (g : b ⟶ c) :
    (f ≫ g).toOrderHom = g.toOrderHom.comp f.toOrderHom := rfl


@[ext]
theorem Hom.ext {a b : SimplexCategory} (f g : a ⟶ b) :
    f.toOrderHom = g.toOrderHom → f = g :=
  Hom.ext' _ _


/-- The constant morphism from [0]. -/
def const (x y : SimplexCategory) (i : Fin (y.len + 1)) : x ⟶ y :=
                            /-
                              x y : SimplexCategory
                              i : Fin (HAdd.hAdd y.len 1)
                              ⊢ Monotone fun x => i
                            -/
  Hom.mk <| ⟨fun _ => i, by tauto⟩
                            /-
                              🎉 no goals
                            -/


@[simp]
                                                /-
                                                  ⊢ Eq ((SimplexCategory.mk 0).const (SimplexCategory.mk 0) 0) (CategoryTheory.C …
                                                -/
lemma const_eq_id : const [0] [0] 0 = 𝟙 _ := by aesop
                                                /-
                                                  🎉 no goals
                                                -/


@[simp]
lemma const_apply (x y : SimplexCategory) (i : Fin (y.len + 1)) (a : Fin (x.len + 1)) :
    (const x y i).toOrderHom a = i := rfl


@[simp]
theorem const_comp (x : SimplexCategory) {y z : SimplexCategory}
    (f : y ⟶ z) (i : Fin (y.len + 1)) :
    const x y i ≫ f = const x z (f.toOrderHom i) :=
  rfl


theorem const_fac_thru_zero (n m : SimplexCategory) (i : Fin (m.len + 1)) :
    const n m i = const n [0] 0 ≫ SimplexCategory.const [0] m i := by
  /-
    n m : SimplexCategory
    i : Fin (HAdd.hAdd m.len 1)
    ⊢ Eq (n.const m i) (CategoryTheory.CategoryStruct.comp (n.const (SimplexCatego …
  -/
  rw [const_comp]; rfl
                   /-
                     🎉 no goals
                   -/


theorem Hom.ext_zero_left {n : SimplexCategory} (f g : ([0] : SimplexCategory) ⟶ n)
    (h0 : f.toOrderHom 0 = g.toOrderHom 0 := by rfl) : f = g := by
  /-
    n : SimplexCategory
    f g : Quiver.Hom (SimplexCategory.mk 0) n
    h0 : autoParam (Eq ((SimplexCategory.Hom.toOrderHom f) 0) ((SimplexCategory.Ho …
    ⊢ Eq f g
  -/
  ext i; match i with | 0 => exact h0 ▸ rfl
         /-
           🎉 no goals
         -/


theorem eq_const_of_zero {n : SimplexCategory} (f : ([0] : SimplexCategory) ⟶ n) :
    f = const _ n (f.toOrderHom 0) := by
  /-
    n : SimplexCategory
    f : Quiver.Hom (SimplexCategory.mk 0) n
    ⊢ Eq f ((SimplexCategory.mk 0).const n ((SimplexCategory.Hom.toOrderHom f) 0))
  -/
  ext x; match x with | 0 => rfl
         /-
           🎉 no goals
         -/


theorem exists_eq_const_of_zero {n : SimplexCategory} (f : ([0] : SimplexCategory) ⟶ n) :
    ∃ a, f = const _ n a := ⟨_, eq_const_of_zero _⟩


theorem eq_const_to_zero {n : SimplexCategory} (f : n ⟶ [0]) :
    f = const n _ 0 := by
  /-
    n : SimplexCategory
    f : Quiver.Hom n (SimplexCategory.mk 0)
    ⊢ Eq f (n.const (SimplexCategory.mk 0) 0)
  -/
  ext : 3
  /-
    case a.h.h
    n : SimplexCategory
    f : Quiver.Hom n (SimplexCategory.mk 0)
    x✝ : Fin (HAdd.hAdd n.len 1)
    ⊢ Eq ((SimplexCategory.Hom.toOrderHom f) x✝) ((SimplexCategory.Hom.toOrderHom  …
  -/
  apply @Subsingleton.elim (Fin 1)
  /-
    🎉 no goals
  -/


theorem Hom.ext_one_left {n : SimplexCategory} (f g : ([1] : SimplexCategory) ⟶ n)
    (h0 : f.toOrderHom 0 = g.toOrderHom 0 := by rfl)
    (h1 : f.toOrderHom 1 = g.toOrderHom 1 := by rfl) : f = g := by
  /-
    n : SimplexCategory
    f g : Quiver.Hom (SimplexCategory.mk 1) n
    h0 : autoParam (Eq ((SimplexCategory.Hom.toOrderHom f) 0) ((SimplexCategory.Ho …
    h1 : autoParam (Eq ((SimplexCategory.Hom.toOrderHom f) 1) ((SimplexCategory.Ho …
    ⊢ Eq f g
  -/
  ext i
  match i with
  | 0 => exact h0 ▸ rfl
  | 1 => exact h1 ▸ rfl


theorem eq_of_one_to_one (f : ([1] : SimplexCategory) ⟶ [1]) :
    (∃ a, f = const [1] _ a) ∨ f = 𝟙 _ := by
  match e0 : f.toOrderHom 0, e1 : f.toOrderHom 1 with
  | 0, 0 | 1, 1 =>
    refine .inl ⟨f.toOrderHom 0, ?_⟩
    ext i : 3
    match i with
    | 0 => rfl
    | 1 => exact e1.trans e0.symm
  | 0, 1 =>
    right
    ext i : 3
    match i with
    | 0 => exact e0
    | 1 => exact e1
  | 1, 0 =>
    have := f.toOrderHom.monotone (by decide : (0 : Fin 2) ≤ 1)
    rw [e0, e1] at this
    exact Not.elim (by decide) this



/-- Make a morphism `[n] ⟶ [m]` from a monotone map between fin's.
This is useful for constructing morphisms between `[n]` directly
without identifying `n` with `[n].len`.
-/
@[simp]
def mkHom {n m : ℕ} (f : Fin (n + 1) →o Fin (m + 1)) : ([n] : SimplexCategory) ⟶ [m] :=
  SimplexCategory.Hom.mk f


/-- The morphism `[1] ⟶ [n]` that picks out a specified `h : i ≤ j` in `Fin (n+1)`.-/
def mkOfLe {n} (i j : Fin (n+1)) (h : i ≤ j) : ([1] : SimplexCategory) ⟶ [n] :=
  SimplexCategory.mkHom {
    toFun := fun | 0 => i | 1 => j
    monotone' := fun
      | 0, 0, _ | 1, 1, _ => le_rfl
      | 0, 1, _ => h
  }


@[simp]
lemma mkOfLe_refl {n} (j : Fin (n + 1)) :
                   /-
                     n : Nat
                     j : Fin (HAdd.hAdd n 1)
                     ⊢ LE.le j j
                   -/
                   /-
                     🎉 no goals
                   -/
                                               /-
                                                 🎉 no goals
                                               -/
    mkOfLe j j (by omega) = [1].const [n] j := Hom.ext_one_left _ _
                                               /-
                                                 🎉 no goals
                                               -/


/-- The morphism `[1] ⟶ [n]` that picks out the "diagonal composite" edge-/
def diag (n : ℕ) : ([1] : SimplexCategory) ⟶ [n] :=
  mkOfLe 0 n (Fin.zero_le _)


/-- The morphism `[1] ⟶ [n]` that picks out the edge spanning the interval from `j` to `j + l`.-/
def intervalEdge {n} (j l : ℕ) (hjl : j + l ≤ n) : ([1] : SimplexCategory) ⟶ [n] :=
                 /-
                   n j l : Nat
                   hjl : LE.le (HAdd.hAdd j l) n
                   ⊢ LT.lt j (HAdd.hAdd n 1)
                 -/
                 /-
                   🎉 no goals
                 -/
  mkOfLe ⟨j, (by omega)⟩ ⟨j + l, (by omega)⟩ (Nat.le_add_right j l)
                                     /-
                                       🎉 no goals
                                     -/


/-- The morphism `[1] ⟶ [n]` that picks out the arrow `i ⟶ i+1` in `Fin (n+1)`.-/
def mkOfSucc {n} (i : Fin n) : ([1] : SimplexCategory) ⟶ [n] :=
  SimplexCategory.mkHom {
    toFun := fun | 0 => i.castSucc | 1 => i.succ
    monotone' := fun
      | 0, 0, _ | 1, 1, _ => le_rfl
      | 0, 1, _ => Fin.castSucc_le_succ i
  }


@[simp]
lemma mkOfSucc_homToOrderHom_zero {n} (i : Fin n) :
    DFunLike.coe (F := Fin 2 →o Fin (n+1)) (Hom.toOrderHom (mkOfSucc i)) 0 = i.castSucc := rfl


@[simp]
lemma mkOfSucc_homToOrderHom_one {n} (i : Fin n) :
    DFunLike.coe (F := Fin 2 →o Fin (n+1)) (Hom.toOrderHom (mkOfSucc i)) 1 = i.succ := rfl



/-- The morphism `[2] ⟶ [n]` that picks out a specified composite of morphisms in `Fin (n+1)`.-/
def mkOfLeComp {n} (i j k : Fin (n + 1)) (h₁ : i ≤ j) (h₂ : j ≤ k) :
    ([2] : SimplexCategory) ⟶ [n] :=
  SimplexCategory.mkHom {
    toFun := fun | 0 => i | 1 => j | 2 => k
    monotone' := fun
      | 0, 0, _ | 1, 1, _ | 2, 2, _  => le_rfl
      | 0, 1, _ => h₁
      | 1, 2, _ => h₂
      | 0, 2, _ => Fin.le_trans h₁ h₂
  }


/-- The "inert" morphism associated to a subinterval `j ≤ i ≤ j + l` of `Fin (n + 1)`.-/
def subinterval {n} (j l : ℕ) (hjl : j + l ≤ n) :
    ([l] : SimplexCategory) ⟶ [n] :=
  SimplexCategory.mkHom {
                                    /-
                                      n j l : Nat
                                      hjl : LE.le (HAdd.hAdd j l) n
                                      i : Fin (HAdd.hAdd l 1)
                                      ⊢ LT.lt (HAdd.hAdd (↑i) j) (HAdd.hAdd n 1)
                                    -/
    toFun := fun i => ⟨i.1 + j, (by omega)⟩
                                    /-
                                      🎉 no goals
                                    -/
                                     /-
                                       n j l : Nat
                                       hjl : LE.le (HAdd.hAdd j l) n
                                       i i' : Fin (HAdd.hAdd l 1)
                                       hii' : LE.le i i'
                                       ⊢ LE.le ((fun i => ⟨HAdd.hAdd (↑i) j, ⋯⟩) i) ((fun i => ⟨HAdd.hAdd (↑i) j, ⋯⟩) …
                                     -/
    monotone' := fun i i' hii' => by simpa only [Fin.mk_le_mk, add_le_add_iff_right] using hii'
                                     /-
                                       🎉 no goals
                                     -/
  }


lemma const_subinterval_eq {n} (j l : ℕ) (hjl : j + l ≤ n) (i : Fin (l + 1)) :
    [0].const [l] i ≫ subinterval j l hjl =
    [0].const [n] ⟨j + i.1, lt_add_of_lt_add_right (Nat.add_lt_add_left i.2 j) hjl⟩  := by
  /-
    n j l : Nat
    hjl : LE.le (HAdd.hAdd j l) n
    i : Fin (HAdd.hAdd l 1)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((SimplexCategory.mk 0).const (Simple …
  -/
  rw [const_comp]
  /-
    n j l : Nat
    hjl : LE.le (HAdd.hAdd j l) n
    i : Fin (HAdd.hAdd l 1)
    ⊢ Eq ((SimplexCategory.mk 0).const (SimplexCategory.mk n) ((SimplexCategory.Ho …
  -/
  congr
  /-
    case e_i
    n j l : Nat
    hjl : LE.le (HAdd.hAdd j l) n
    i : Fin (HAdd.hAdd l 1)
    ⊢ Eq ((SimplexCategory.Hom.toOrderHom (SimplexCategory.subinterval j l hjl)) i …
  -/
  ext
  /-
    case e_i.h
    n j l : Nat
    hjl : LE.le (HAdd.hAdd j l) n
    i : Fin (HAdd.hAdd l 1)
    ⊢ Eq ↑((SimplexCategory.Hom.toOrderHom (SimplexCategory.subinterval j l hjl))  …
  -/
  dsimp [subinterval]
  /-
    case e_i.h
    n j l : Nat
    hjl : LE.le (HAdd.hAdd j l) n
    i : Fin (HAdd.hAdd l 1)
    ⊢ Eq (HAdd.hAdd (↑i) j) (HAdd.hAdd j ↑i)
  -/
  rw [add_comm]
  /-
    🎉 no goals
  -/


@[simp]
lemma mkOfSucc_subinterval_eq {n} (j l : ℕ) (hjl : j + l ≤ n) (i : Fin l) :
    mkOfSucc i ≫ subinterval j l hjl =
    mkOfSucc ⟨j + i.1, Nat.lt_of_lt_of_le (Nat.add_lt_add_left i.2 j) hjl⟩ := by
  /-
    n j l : Nat
    hjl : LE.le (HAdd.hAdd j l) n
    i : Fin l
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (SimplexCategory.mkOfSucc i) (Simplex …
  -/
  unfold subinterval mkOfSucc
  /-
    n j l : Nat
    hjl : LE.le (HAdd.hAdd j l) n
    i : Fin l
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (SimplexCategory.mkHom { toFun := fun …
  -/
  ext i
  match i with
  | 0 =>
    simp only [len_mk, Nat.reduceAdd, mkHom, comp_toOrderHom, Hom.toOrderHom_mk,
      OrderHom.mk_comp_mk, Fin.isValue, OrderHom.coe_mk, Function.comp_apply, Fin.castSucc_mk,
      Fin.succ_mk]
    rw [add_comm]
    rfl
  | 1 =>
    simp only [len_mk, Nat.reduceAdd, mkHom, comp_toOrderHom, Hom.toOrderHom_mk,
      OrderHom.mk_comp_mk, Fin.isValue, OrderHom.coe_mk, Function.comp_apply, Fin.castSucc_mk,
      Fin.succ_mk]
    rw [← Nat.add_comm j _]
    rfl


@[simp]
lemma diag_subinterval_eq {n} (j l : ℕ) (hjl : j + l ≤ n) :
    diag l ≫ subinterval j l hjl = intervalEdge j l hjl := by
  /-
    n j l : Nat
    hjl : LE.le (HAdd.hAdd j l) n
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (SimplexCategory.diag l) (SimplexCate …
  -/
  unfold subinterval intervalEdge diag mkOfLe
  /-
    n j l : Nat
    hjl : LE.le (HAdd.hAdd j l) n
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (SimplexCategory.mkHom { toFun := fun …
  -/
  ext i
  match i with
  | 0 =>
    simp only [len_mk, Nat.reduceAdd, mkHom, Fin.natCast_eq_last, comp_toOrderHom,
      Hom.toOrderHom_mk, OrderHom.mk_comp_mk, Fin.isValue, OrderHom.coe_mk, Function.comp_apply]
    rw [Nat.add_comm]
    rfl
  | 1 =>
    simp only [len_mk, Nat.reduceAdd, mkHom, Fin.natCast_eq_last, comp_toOrderHom,
      Hom.toOrderHom_mk, OrderHom.mk_comp_mk, Fin.isValue, OrderHom.coe_mk, Function.comp_apply]
    rw [Nat.add_comm]
    rfl


instance (Δ : SimplexCategory) : Subsingleton (Δ ⟶ [0]) where
                  /-
                    Δ : SimplexCategory
                    f g : Quiver.Hom Δ (SimplexCategory.mk 0)
                    ⊢ Eq f g
                  -/
  allEq f g := by ext : 3; apply Subsingleton.elim (α := Fin 1)
                           /-
                             🎉 no goals
                           -/


theorem hom_zero_zero (f : ([0] : SimplexCategory) ⟶ [0]) : f = 𝟙 _ := by
  /-
    f : Quiver.Hom (SimplexCategory.mk 0) (SimplexCategory.mk 0)
    ⊢ Eq f (CategoryTheory.CategoryStruct.id (SimplexCategory.mk 0))
  -/
  apply Subsingleton.elim
  /-
    🎉 no goals
  -/


/-- The `i`-th face map from `[n]` to `[n+1]` -/
def δ {n} (i : Fin (n + 2)) : ([n] : SimplexCategory) ⟶ [n + 1] :=
  mkHom (Fin.succAboveOrderEmb i).toOrderHom


/-- The `i`-th degeneracy map from `[n+1]` to `[n]` -/
def σ {n} (i : Fin (n + 1)) : ([n + 1] : SimplexCategory) ⟶ [n] :=
  mkHom
    { toFun := Fin.predAbove i
      monotone' := Fin.predAbove_right_monotone i }


/-- The generic case of the first simplicial identity -/
theorem δ_comp_δ {n} {i j : Fin (n + 2)} (H : i ≤ j) :
    δ i ≫ δ j.succ = δ j ≫ δ i.castSucc := by
  /-
    n : Nat
    i j : Fin (HAdd.hAdd n 2)
    H : LE.le i j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (SimplexCategory.δ i) (SimplexCategor …
  -/
  ext k
  /-
    case a.h.h.h
    n : Nat
    i j : Fin (HAdd.hAdd n 2)
    H : LE.le i j
    k : Fin (HAdd.hAdd (SimplexCategory.mk n).len 1)
    ⊢ Eq ↑((SimplexCategory.Hom.toOrderHom (CategoryTheory.CategoryStruct.comp (Si …
  -/
  dsimp [δ, Fin.succAbove]
  /-
    case a.h.h.h
    n : Nat
    i j : Fin (HAdd.hAdd n 2)
    H : LE.le i j
    k : Fin (HAdd.hAdd (SimplexCategory.mk n).len 1)
    ⊢ Eq ↑(ite (LT.lt (ite (LT.lt k.castSucc i) k.castSucc k.succ).castSucc j.succ …
  -/
  rcases i with ⟨i, _⟩
  /-
    case a.h.h.h.mk
    n : Nat
    j : Fin (HAdd.hAdd n 2)
    k : Fin (HAdd.hAdd (SimplexCategory.mk n).len 1)
    i : Nat
    isLt✝ : LT.lt i (HAdd.hAdd n 2)
    H : LE.le ⟨i, isLt✝⟩ j
    ⊢ Eq ↑(ite (LT.lt (ite (LT.lt k.castSucc ⟨i, isLt✝⟩) k.castSucc k.succ).castSu …
  -/
  rcases j with ⟨j, _⟩
  /-
    case a.h.h.h.mk.mk
    n : Nat
    k : Fin (HAdd.hAdd (SimplexCategory.mk n).len 1)
    i : Nat
    isLt✝¹ : LT.lt i (HAdd.hAdd n 2)
    j : Nat
    isLt✝ : LT.lt j (HAdd.hAdd n 2)
    H : LE.le ⟨i, isLt✝¹⟩ ⟨j, isLt✝⟩
    ⊢ Eq ↑(ite (LT.lt (ite (LT.lt k.castSucc ⟨i, isLt✝¹⟩) k.castSucc k.succ).castS …
  -/
  rcases k with ⟨k, _⟩
  /-
    case a.h.h.h.mk.mk.mk
    n i : Nat
    isLt✝² : LT.lt i (HAdd.hAdd n 2)
    j : Nat
    isLt✝¹ : LT.lt j (HAdd.hAdd n 2)
    H : LE.le ⟨i, isLt✝²⟩ ⟨j, isLt✝¹⟩
    k : Nat
    isLt✝ : LT.lt k (HAdd.hAdd (SimplexCategory.mk n).len 1)
    ⊢ Eq ↑(ite (LT.lt (ite (LT.lt ⟨k, isLt✝⟩.castSucc ⟨i, isLt✝²⟩) ⟨k, isLt✝⟩.cast …
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
  split_ifs <;> · simp at * <;> omega
                  /-
                    🎉 no goals
                  -/


theorem δ_comp_δ' {n} {i : Fin (n + 2)} {j : Fin (n + 3)} (H : i.castSucc < j) :
    δ i ≫ δ j =
                                       /-
                                         n : Nat
                                         i : Fin (HAdd.hAdd n 2)
                                         j : Fin (HAdd.hAdd n 3)
                                         H : LT.lt i.castSucc j
                                         hj : Eq j 0
                                         ⊢ False
                                       -/
      δ (j.pred fun (hj : j = 0) => by simp [hj, Fin.not_lt_zero] at H) ≫
                                       /-
                                         🎉 no goals
                                       -/
        δ (Fin.castSucc i) := by
  /-
    n : Nat
    i : Fin (HAdd.hAdd n 2)
    j : Fin (HAdd.hAdd n 3)
    H : LT.lt i.castSucc j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (SimplexCategory.δ i) (SimplexCategor …
  -/
  rw [← δ_comp_δ]
    /-
      n : Nat
      i : Fin (HAdd.hAdd n 2)
      j : Fin (HAdd.hAdd n 3)
      H : LT.lt i.castSucc j
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (SimplexCategory.δ i) (SimplexCategor …
    -/
  · rw [Fin.succ_pred]
    /-
      🎉 no goals
    -/
  · simpa only [Fin.le_iff_val_le_val, ← Nat.lt_succ_iff, Nat.succ_eq_add_one, ← Fin.val_succ,
      j.succ_pred, Fin.lt_iff_val_lt_val] using H


theorem δ_comp_δ'' {n} {i : Fin (n + 3)} {j : Fin (n + 2)} (H : i ≤ Fin.castSucc j) :
    δ (i.castLT (Nat.lt_of_le_of_lt (Fin.le_iff_val_le_val.mp H) j.is_lt)) ≫ δ j.succ =
      δ j ≫ δ i := by
  /-
    n : Nat
    i : Fin (HAdd.hAdd n 3)
    j : Fin (HAdd.hAdd n 2)
    H : LE.le i j.castSucc
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (SimplexCategory.δ (i.castLT ⋯)) (Sim …
  -/
  rw [δ_comp_δ]
    /-
      n : Nat
      i : Fin (HAdd.hAdd n 3)
      j : Fin (HAdd.hAdd n 2)
      H : LE.le i j.castSucc
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (SimplexCategory.δ j) (SimplexCategor …
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      n : Nat
      i : Fin (HAdd.hAdd n 3)
      j : Fin (HAdd.hAdd n 2)
      H : LE.le i j.castSucc
      ⊢ LE.le (i.castLT ⋯) j
    -/
  · exact H
    /-
      🎉 no goals
    -/


/-- The special case of the first simplicial identity -/
@[reassoc]
theorem δ_comp_δ_self {n} {i : Fin (n + 2)} : δ i ≫ δ i.castSucc = δ i ≫ δ i.succ :=
  (δ_comp_δ (le_refl i)).symm


@[reassoc]
theorem δ_comp_δ_self' {n} {i : Fin (n + 2)} {j : Fin (n + 3)} (H : j = i.castSucc) :
    δ i ≫ δ j = δ i ≫ δ i.succ := by
  /-
    n : Nat
    i : Fin (HAdd.hAdd n 2)
    j : Fin (HAdd.hAdd n 3)
    H : Eq j i.castSucc
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (SimplexCategory.δ i) (SimplexCategor …
  -/
  subst H
  /-
    n : Nat
    i : Fin (HAdd.hAdd n 2)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (SimplexCategory.δ i) (SimplexCategor …
  -/
  rw [δ_comp_δ_self]
  /-
    🎉 no goals
  -/


/-- The second simplicial identity -/
@[reassoc]
theorem δ_comp_σ_of_le {n} {i : Fin (n + 2)} {j : Fin (n + 1)} (H : i ≤ j.castSucc) :
    δ i.castSucc ≫ σ j.succ = σ j ≫ δ i := by
  /-
    n : Nat
    i : Fin (HAdd.hAdd n 2)
    j : Fin (HAdd.hAdd n 1)
    H : LE.le i j.castSucc
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (SimplexCategory.δ i.castSucc) (Simpl …
  -/
  ext k : 3
  /-
    case a.h.h
    n : Nat
    i : Fin (HAdd.hAdd n 2)
    j : Fin (HAdd.hAdd n 1)
    H : LE.le i j.castSucc
    k : Fin (HAdd.hAdd (SimplexCategory.mk (HAdd.hAdd n 1)).len 1)
    ⊢ Eq ((SimplexCategory.Hom.toOrderHom (CategoryTheory.CategoryStruct.comp (Sim …
  -/
  dsimp [σ, δ]
  /-
    case a.h.h
    n : Nat
    i : Fin (HAdd.hAdd n 2)
    j : Fin (HAdd.hAdd n 1)
    H : LE.le i j.castSucc
    k : Fin (HAdd.hAdd (SimplexCategory.mk (HAdd.hAdd n 1)).len 1)
    ⊢ Eq (j.succ.predAbove (i.castSucc.succAbove k)) (i.succAbove (j.predAbove k))
  -/
  rcases le_or_lt i k with (hik | hik)
  · rw [Fin.succAbove_of_le_castSucc _ _ (Fin.castSucc_le_castSucc_iff.mpr hik),
    Fin.succ_predAbove_succ, Fin.succAbove_of_le_castSucc]
    /-
      case a.h.h.inl.h
      n : Nat
      i : Fin (HAdd.hAdd n 2)
      j : Fin (HAdd.hAdd n 1)
      H : LE.le i j.castSucc
      k : Fin (HAdd.hAdd (SimplexCategory.mk (HAdd.hAdd n 1)).len 1)
      hik : LE.le i k
      ⊢ LE.le i (j.predAbove k).castSucc
    -/
    rcases le_or_lt k (j.castSucc) with (hjk | hjk)
      /-
        case a.h.h.inl.h.inl
        n : Nat
        i : Fin (HAdd.hAdd n 2)
        j : Fin (HAdd.hAdd n 1)
        H : LE.le i j.castSucc
        k : Fin (HAdd.hAdd (SimplexCategory.mk (HAdd.hAdd n 1)).len 1)
        hik : LE.le i k
        hjk : LE.le k j.castSucc
        ⊢ LE.le i (j.predAbove k).castSucc
      -/
    · rwa [Fin.predAbove_of_le_castSucc _ _ hjk, Fin.castSucc_castPred]
      /-
        🎉 no goals
      -/
      /-
        case a.h.h.inl.h.inr
        n : Nat
        i : Fin (HAdd.hAdd n 2)
        j : Fin (HAdd.hAdd n 1)
        H : LE.le i j.castSucc
        k : Fin (HAdd.hAdd (SimplexCategory.mk (HAdd.hAdd n 1)).len 1)
        hik : LE.le i k
        hjk : LT.lt j.castSucc k
        ⊢ LE.le i (j.predAbove k).castSucc
      -/
    · rw [Fin.le_castSucc_iff, Fin.predAbove_of_castSucc_lt _ _ hjk, Fin.succ_pred]
      /-
        case a.h.h.inl.h.inr
        n : Nat
        i : Fin (HAdd.hAdd n 2)
        j : Fin (HAdd.hAdd n 1)
        H : LE.le i j.castSucc
        k : Fin (HAdd.hAdd (SimplexCategory.mk (HAdd.hAdd n 1)).len 1)
        hik : LE.le i k
        hjk : LT.lt j.castSucc k
        ⊢ LT.lt i k
      -/
      exact H.trans_lt hjk
      /-
        🎉 no goals
      -/
    /-
      case a.h.h.inr
      n : Nat
      i : Fin (HAdd.hAdd n 2)
      j : Fin (HAdd.hAdd n 1)
      H : LE.le i j.castSucc
      k : Fin (HAdd.hAdd (SimplexCategory.mk (HAdd.hAdd n 1)).len 1)
      hik : LT.lt k i
      ⊢ Eq (j.succ.predAbove (i.castSucc.succAbove k)) (i.succAbove (j.predAbove k))
    -/
  · rw [Fin.succAbove_of_castSucc_lt _ _ (Fin.castSucc_lt_castSucc_iff.mpr hik)]
    /-
      case a.h.h.inr
      n : Nat
      i : Fin (HAdd.hAdd n 2)
      j : Fin (HAdd.hAdd n 1)
      H : LE.le i j.castSucc
      k : Fin (HAdd.hAdd (SimplexCategory.mk (HAdd.hAdd n 1)).len 1)
      hik : LT.lt k i
      ⊢ Eq (j.succ.predAbove k.castSucc) (i.succAbove (j.predAbove k))
    -/
    have hjk := H.trans_lt' hik
    rw [Fin.predAbove_of_le_castSucc _ _ (Fin.castSucc_le_castSucc_iff.mpr
      (hjk.trans (Fin.castSucc_lt_succ _)).le),
      Fin.predAbove_of_le_castSucc _ _ hjk.le, Fin.castPred_castSucc, Fin.succAbove_of_castSucc_lt,
      Fin.castSucc_castPred]
    /-
      case a.h.h.inr.h
      n : Nat
      i : Fin (HAdd.hAdd n 2)
      j : Fin (HAdd.hAdd n 1)
      H : LE.le i j.castSucc
      k : Fin (HAdd.hAdd (SimplexCategory.mk (HAdd.hAdd n 1)).len 1)
      hik : LT.lt k i
      hjk : LT.lt k j.castSucc
      ⊢ LT.lt (k.castPred ⋯).castSucc i
    -/
    rwa [Fin.castSucc_castPred]
    /-
      🎉 no goals
    -/


/-- The first part of the third simplicial identity -/
@[reassoc]
theorem δ_comp_σ_self {n} {i : Fin (n + 1)} :
    δ (Fin.castSucc i) ≫ σ i = 𝟙 ([n] : SimplexCategory) := by
  /-
    n : Nat
    i : Fin (HAdd.hAdd n 1)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (SimplexCategory.δ i.castSucc) (Simpl …
  -/
  rcases i with ⟨i, hi⟩
  /-
    case mk
    n i : Nat
    hi : LT.lt i (HAdd.hAdd n 1)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (SimplexCategory.δ ⟨i, hi⟩.castSucc)  …
  -/
  ext ⟨j, hj⟩
  /-
    case mk.a.h.h.mk.h
    n i : Nat
    hi : LT.lt i (HAdd.hAdd n 1)
    j : Nat
    hj : LT.lt j (HAdd.hAdd (SimplexCategory.mk n).len 1)
    ⊢ Eq ↑((SimplexCategory.Hom.toOrderHom (CategoryTheory.CategoryStruct.comp (Si …
  -/
  simp? at hj says simp only [len_mk] at hj
  /-
    case mk.a.h.h.mk.h
    n i : Nat
    hi : LT.lt i (HAdd.hAdd n 1)
    j : Nat
    hj : LT.lt j (HAdd.hAdd n 1)
    ⊢ Eq ↑((SimplexCategory.Hom.toOrderHom (CategoryTheory.CategoryStruct.comp (Si …
  -/
  dsimp [σ, δ, Fin.predAbove, Fin.succAbove]
  /-
    case mk.a.h.h.mk.h
    n i : Nat
    hi : LT.lt i (HAdd.hAdd n 1)
    j : Nat
    hj : LT.lt j (HAdd.hAdd n 1)
    ⊢ Eq (↑(dite (LT.lt ⟨i, ⋯⟩ (ite (LT.lt ⟨j, ⋯⟩ ⟨i, ⋯⟩) ⟨j, ⋯⟩ ⟨HAdd.hAdd j 1, ⋯ …
  -/
  simp only [Fin.lt_iff_val_lt_val, Fin.dite_val, Fin.ite_val, Fin.coe_pred, Fin.coe_castLT]
  /-
    case mk.a.h.h.mk.h
    n i : Nat
    hi : LT.lt i (HAdd.hAdd n 1)
    j : Nat
    hj : LT.lt j (HAdd.hAdd n 1)
    ⊢ Eq (dite (LT.lt i (ite (LT.lt j i) j (HAdd.hAdd j 1))) (fun h => HSub.hSub ( …
  -/
  split_ifs
  /-
    case pos
    n i : Nat
    hi : LT.lt i (HAdd.hAdd n 1)
    j : Nat
    hj : LT.lt j (HAdd.hAdd n 1)
    h✝¹ : LT.lt j i
    h✝ : LT.lt i j
    ⊢ Eq (HSub.hSub j 1) j
  -/
  any_goals simp
  /-
    case pos
    n i : Nat
    hi : LT.lt i (HAdd.hAdd n 1)
    j : Nat
    hj : LT.lt j (HAdd.hAdd n 1)
    h✝¹ : LT.lt j i
    h✝ : LT.lt i j
    ⊢ Eq (HSub.hSub j 1) j
  -/
  all_goals omega
  /-
    🎉 no goals
  -/


@[reassoc]
theorem δ_comp_σ_self' {n} {j : Fin (n + 2)} {i : Fin (n + 1)} (H : j = i.castSucc) :
    δ j ≫ σ i = 𝟙 ([n] : SimplexCategory) := by
  /-
    n : Nat
    j : Fin (HAdd.hAdd n 2)
    i : Fin (HAdd.hAdd n 1)
    H : Eq j i.castSucc
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (SimplexCategory.δ j) (SimplexCategor …
  -/
  subst H
  /-
    n : Nat
    i : Fin (HAdd.hAdd n 1)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (SimplexCategory.δ i.castSucc) (Simpl …
  -/
  rw [δ_comp_σ_self]
  /-
    🎉 no goals
  -/


/-- The second part of the third simplicial identity -/
@[reassoc]
theorem δ_comp_σ_succ {n} {i : Fin (n + 1)} : δ i.succ ≫ σ i = 𝟙 ([n] : SimplexCategory) := by
  /-
    n : Nat
    i : Fin (HAdd.hAdd n 1)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (SimplexCategory.δ i.succ) (SimplexCa …
  -/
  ext j
  /-
    case a.h.h.h
    n : Nat
    i : Fin (HAdd.hAdd n 1)
    j : Fin (HAdd.hAdd (SimplexCategory.mk n).len 1)
    ⊢ Eq ↑((SimplexCategory.Hom.toOrderHom (CategoryTheory.CategoryStruct.comp (Si …
  -/
  rcases i with ⟨i, _⟩
  /-
    case a.h.h.h.mk
    n : Nat
    j : Fin (HAdd.hAdd (SimplexCategory.mk n).len 1)
    i : Nat
    isLt✝ : LT.lt i (HAdd.hAdd n 1)
    ⊢ Eq ↑((SimplexCategory.Hom.toOrderHom (CategoryTheory.CategoryStruct.comp (Si …
  -/
  rcases j with ⟨j, _⟩
  /-
    case a.h.h.h.mk.mk
    n i : Nat
    isLt✝¹ : LT.lt i (HAdd.hAdd n 1)
    j : Nat
    isLt✝ : LT.lt j (HAdd.hAdd (SimplexCategory.mk n).len 1)
    ⊢ Eq ↑((SimplexCategory.Hom.toOrderHom (CategoryTheory.CategoryStruct.comp (Si …
  -/
  dsimp [δ, σ, Fin.succAbove, Fin.predAbove]
  /-
    case a.h.h.h.mk.mk
    n i : Nat
    isLt✝¹ : LT.lt i (HAdd.hAdd n 1)
    j : Nat
    isLt✝ : LT.lt j (HAdd.hAdd (SimplexCategory.mk n).len 1)
    ⊢ Eq (↑(dite (LT.lt ⟨i, ⋯⟩ (ite (LT.lt ⟨j, ⋯⟩ ⟨HAdd.hAdd i 1, ⋯⟩) ⟨j, ⋯⟩ ⟨HAdd …
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
  split_ifs <;> simp <;> simp at * <;> omega
                                       /-
                                         🎉 no goals
                                       -/


@[reassoc]
theorem δ_comp_σ_succ' {n} {j : Fin (n + 2)} {i : Fin (n + 1)} (H : j = i.succ) :
    δ j ≫ σ i = 𝟙 ([n] : SimplexCategory) := by
  /-
    n : Nat
    j : Fin (HAdd.hAdd n 2)
    i : Fin (HAdd.hAdd n 1)
    H : Eq j i.succ
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (SimplexCategory.δ j) (SimplexCategor …
  -/
  subst H
  /-
    n : Nat
    i : Fin (HAdd.hAdd n 1)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (SimplexCategory.δ i.succ) (SimplexCa …
  -/
  rw [δ_comp_σ_succ]
  /-
    🎉 no goals
  -/


/-- The fourth simplicial identity -/
@[reassoc]
theorem δ_comp_σ_of_gt {n} {i : Fin (n + 2)} {j : Fin (n + 1)} (H : j.castSucc < i) :
    δ i.succ ≫ σ j.castSucc = σ j ≫ δ i := by
  /-
    n : Nat
    i : Fin (HAdd.hAdd n 2)
    j : Fin (HAdd.hAdd n 1)
    H : LT.lt j.castSucc i
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (SimplexCategory.δ i.succ) (SimplexCa …
  -/
  ext k : 3
  /-
    case a.h.h
    n : Nat
    i : Fin (HAdd.hAdd n 2)
    j : Fin (HAdd.hAdd n 1)
    H : LT.lt j.castSucc i
    k : Fin (HAdd.hAdd (SimplexCategory.mk (HAdd.hAdd n 1)).len 1)
    ⊢ Eq ((SimplexCategory.Hom.toOrderHom (CategoryTheory.CategoryStruct.comp (Sim …
  -/
  dsimp [δ, σ]
  /-
    case a.h.h
    n : Nat
    i : Fin (HAdd.hAdd n 2)
    j : Fin (HAdd.hAdd n 1)
    H : LT.lt j.castSucc i
    k : Fin (HAdd.hAdd (SimplexCategory.mk (HAdd.hAdd n 1)).len 1)
    ⊢ Eq (j.castSucc.predAbove (i.succ.succAbove k)) (i.succAbove (j.predAbove k))
  -/
  rcases le_or_lt k i with (hik | hik)
    /-
      case a.h.h.inl
      n : Nat
      i : Fin (HAdd.hAdd n 2)
      j : Fin (HAdd.hAdd n 1)
      H : LT.lt j.castSucc i
      k : Fin (HAdd.hAdd (SimplexCategory.mk (HAdd.hAdd n 1)).len 1)
      hik : LE.le k i
      ⊢ Eq (j.castSucc.predAbove (i.succ.succAbove k)) (i.succAbove (j.predAbove k))
    -/
  · rw [Fin.succAbove_of_castSucc_lt _ _ (Fin.castSucc_lt_succ_iff.mpr hik)]
    /-
      case a.h.h.inl
      n : Nat
      i : Fin (HAdd.hAdd n 2)
      j : Fin (HAdd.hAdd n 1)
      H : LT.lt j.castSucc i
      k : Fin (HAdd.hAdd (SimplexCategory.mk (HAdd.hAdd n 1)).len 1)
      hik : LE.le k i
      ⊢ Eq (j.castSucc.predAbove k.castSucc) (i.succAbove (j.predAbove k))
    -/
    rcases le_or_lt k (j.castSucc) with (hjk | hjk)
    · rw [Fin.predAbove_of_le_castSucc _ _
      (Fin.castSucc_le_castSucc_iff.mpr hjk), Fin.castPred_castSucc,
      Fin.predAbove_of_le_castSucc _ _ hjk, Fin.succAbove_of_castSucc_lt, Fin.castSucc_castPred]
      /-
        case a.h.h.inl.inl.h
        n : Nat
        i : Fin (HAdd.hAdd n 2)
        j : Fin (HAdd.hAdd n 1)
        H : LT.lt j.castSucc i
        k : Fin (HAdd.hAdd (SimplexCategory.mk (HAdd.hAdd n 1)).len 1)
        hik : LE.le k i
        hjk : LE.le k j.castSucc
        ⊢ LT.lt (k.castPred ⋯).castSucc i
      -/
      rw [Fin.castSucc_castPred]
      /-
        case a.h.h.inl.inl.h
        n : Nat
        i : Fin (HAdd.hAdd n 2)
        j : Fin (HAdd.hAdd n 1)
        H : LT.lt j.castSucc i
        k : Fin (HAdd.hAdd (SimplexCategory.mk (HAdd.hAdd n 1)).len 1)
        hik : LE.le k i
        hjk : LE.le k j.castSucc
        ⊢ LT.lt k i
      -/
      exact hjk.trans_lt H
      /-
        🎉 no goals
      -/
    · rw [Fin.predAbove_of_castSucc_lt _ _ (Fin.castSucc_lt_castSucc_iff.mpr hjk),
      Fin.predAbove_of_castSucc_lt _ _ hjk, Fin.succAbove_of_castSucc_lt,
      Fin.castSucc_pred_eq_pred_castSucc]
      /-
        case a.h.h.inl.inr.h
        n : Nat
        i : Fin (HAdd.hAdd n 2)
        j : Fin (HAdd.hAdd n 1)
        H : LT.lt j.castSucc i
        k : Fin (HAdd.hAdd (SimplexCategory.mk (HAdd.hAdd n 1)).len 1)
        hik : LE.le k i
        hjk : LT.lt j.castSucc k
        ⊢ LT.lt (k.pred ⋯).castSucc i
      -/
      rwa [Fin.castSucc_lt_iff_succ_le, Fin.succ_pred]
      /-
        🎉 no goals
      -/
    /-
      case a.h.h.inr
      n : Nat
      i : Fin (HAdd.hAdd n 2)
      j : Fin (HAdd.hAdd n 1)
      H : LT.lt j.castSucc i
      k : Fin (HAdd.hAdd (SimplexCategory.mk (HAdd.hAdd n 1)).len 1)
      hik : LT.lt i k
      ⊢ Eq (j.castSucc.predAbove (i.succ.succAbove k)) (i.succAbove (j.predAbove k))
    -/
  · rw [Fin.succAbove_of_le_castSucc _ _ (Fin.succ_le_castSucc_iff.mpr hik)]
    /-
      case a.h.h.inr
      n : Nat
      i : Fin (HAdd.hAdd n 2)
      j : Fin (HAdd.hAdd n 1)
      H : LT.lt j.castSucc i
      k : Fin (HAdd.hAdd (SimplexCategory.mk (HAdd.hAdd n 1)).len 1)
      hik : LT.lt i k
      ⊢ Eq (j.castSucc.predAbove k.succ) (i.succAbove (j.predAbove k))
    -/
    have hjk := H.trans hik
    rw [Fin.predAbove_of_castSucc_lt _ _ hjk, Fin.predAbove_of_castSucc_lt _ _
      (Fin.castSucc_lt_succ_iff.mpr hjk.le),
    Fin.pred_succ, Fin.succAbove_of_le_castSucc, Fin.succ_pred]
    /-
      case a.h.h.inr.h
      n : Nat
      i : Fin (HAdd.hAdd n 2)
      j : Fin (HAdd.hAdd n 1)
      H : LT.lt j.castSucc i
      k : Fin (HAdd.hAdd (SimplexCategory.mk (HAdd.hAdd n 1)).len 1)
      hik : LT.lt i k
      hjk : LT.lt j.castSucc k
      ⊢ LE.le i (k.pred ⋯).castSucc
    -/
    rwa [Fin.le_castSucc_pred_iff]
    /-
      🎉 no goals
    -/


@[reassoc]
theorem δ_comp_σ_of_gt' {n} {i : Fin (n + 3)} {j : Fin (n + 2)} (H : j.succ < i) :
    δ i ≫ σ j = σ (j.castLT ((add_lt_add_iff_right 1).mp (lt_of_lt_of_le H i.is_le))) ≫
                                       /-
                                         n : Nat
                                         i : Fin (HAdd.hAdd n 3)
                                         j : Fin (HAdd.hAdd n 2)
                                         H : LT.lt j.succ i
                                         hi : Eq i 0
                                         ⊢ False
                                       -/
      δ (i.pred fun (hi : i = 0) => by simp only [Fin.not_lt_zero, hi] at H) := by
                                       /-
                                         🎉 no goals
                                       -/
  /-
    n : Nat
    i : Fin (HAdd.hAdd n 3)
    j : Fin (HAdd.hAdd n 2)
    H : LT.lt j.succ i
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (SimplexCategory.δ i) (SimplexCategor …
  -/
  rw [← δ_comp_σ_of_gt]
    /-
      n : Nat
      i : Fin (HAdd.hAdd n 3)
      j : Fin (HAdd.hAdd n 2)
      H : LT.lt j.succ i
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (SimplexCategory.δ i) (SimplexCategor …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      n : Nat
      i : Fin (HAdd.hAdd n 3)
      j : Fin (HAdd.hAdd n 2)
      H : LT.lt j.succ i
      ⊢ LT.lt (j.castLT ⋯).castSucc (i.pred ⋯)
    -/
  · rw [Fin.castSucc_castLT, ← Fin.succ_lt_succ_iff, Fin.succ_pred]
    /-
      n : Nat
      i : Fin (HAdd.hAdd n 3)
      j : Fin (HAdd.hAdd n 2)
      H : LT.lt j.succ i
      ⊢ LT.lt j.succ i
    -/
    exact H
    /-
      🎉 no goals
    -/


/-- The fifth simplicial identity -/
@[reassoc]
theorem σ_comp_σ {n} {i j : Fin (n + 1)} (H : i ≤ j) :
    σ (Fin.castSucc i) ≫ σ j = σ j.succ ≫ σ i := by
  /-
    n : Nat
    i j : Fin (HAdd.hAdd n 1)
    H : LE.le i j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (SimplexCategory.σ i.castSucc) (Simpl …
  -/
  ext k : 3
  /-
    case a.h.h
    n : Nat
    i j : Fin (HAdd.hAdd n 1)
    H : LE.le i j
    k : Fin (HAdd.hAdd (SimplexCategory.mk (HAdd.hAdd (HAdd.hAdd n 1) 1)).len 1)
    ⊢ Eq ((SimplexCategory.Hom.toOrderHom (CategoryTheory.CategoryStruct.comp (Sim …
  -/
  dsimp [σ]
  /-
    case a.h.h
    n : Nat
    i j : Fin (HAdd.hAdd n 1)
    H : LE.le i j
    k : Fin (HAdd.hAdd (SimplexCategory.mk (HAdd.hAdd (HAdd.hAdd n 1) 1)).len 1)
    ⊢ Eq (j.predAbove (i.castSucc.predAbove k)) (i.predAbove (j.succ.predAbove k))
  -/
  cases' k using Fin.lastCases with k
    /-
      case a.h.h.last
      n : Nat
      i j : Fin (HAdd.hAdd n 1)
      H : LE.le i j
      ⊢ Eq (j.predAbove (i.castSucc.predAbove (Fin.last (SimplexCategory.mk (HAdd.hA …
    -/
  · simp only [len_mk, Fin.predAbove_right_last]
    /-
      🎉 no goals
    -/
    /-
      case a.h.h.cast
      n : Nat
      i j : Fin (HAdd.hAdd n 1)
      H : LE.le i j
      k : Fin (SimplexCategory.mk (HAdd.hAdd (HAdd.hAdd n 1) 1)).len
      ⊢ Eq (j.predAbove (i.castSucc.predAbove k.castSucc)) (i.predAbove (j.succ.pred …
    -/
  · cases' k using Fin.cases with k
    · rw [Fin.castSucc_zero, Fin.predAbove_of_le_castSucc _ 0 (Fin.zero_le _),
      Fin.predAbove_of_le_castSucc _ _ (Fin.zero_le _), Fin.castPred_zero,
      Fin.predAbove_of_le_castSucc _ 0 (Fin.zero_le _),
      Fin.predAbove_of_le_castSucc _ _ (Fin.zero_le _)]
      /-
        case a.h.h.cast.succ
        n : Nat
        i j : Fin (HAdd.hAdd n 1)
        H : LE.le i j
        k : Fin (HAdd.hAdd n 1)
        ⊢ Eq (j.predAbove (i.castSucc.predAbove k.succ.castSucc)) (i.predAbove (j.succ …
      -/
    · rcases le_or_lt i k with (h | h)
      · simp_rw [Fin.predAbove_of_castSucc_lt i.castSucc _ (Fin.castSucc_lt_castSucc_iff.mpr
        (Fin.castSucc_lt_succ_iff.mpr h)), ← Fin.succ_castSucc, Fin.pred_succ,
        Fin.succ_predAbove_succ]
        /-
          case a.h.h.cast.succ.inl
          n : Nat
          i j : Fin (HAdd.hAdd n 1)
          H : LE.le i j
          k : Fin (HAdd.hAdd n 1)
          h : LE.le i k
          ⊢ Eq (j.predAbove k.castSucc) (i.predAbove (j.predAbove k.castSucc).succ)
        -/
        rw [Fin.predAbove_of_castSucc_lt i _ (Fin.castSucc_lt_succ_iff.mpr _), Fin.pred_succ]
        /-
          n : Nat
          i j : Fin (HAdd.hAdd n 1)
          H : LE.le i j
          k : Fin (HAdd.hAdd n 1)
          h : LE.le i k
          ⊢ LE.le i (j.predAbove k.castSucc)
        -/
        rcases le_or_lt k j with (hkj | hkj)
        · rwa [Fin.predAbove_of_le_castSucc _ _ (Fin.castSucc_le_castSucc_iff.mpr hkj),
          Fin.castPred_castSucc]
        · rw [Fin.predAbove_of_castSucc_lt _ _ (Fin.castSucc_lt_castSucc_iff.mpr hkj),
          Fin.le_pred_iff,
          Fin.succ_le_castSucc_iff]
          /-
            case inr
            n : Nat
            i j : Fin (HAdd.hAdd n 1)
            H : LE.le i j
            k : Fin (HAdd.hAdd n 1)
            h : LE.le i k
            hkj : LT.lt j k
            ⊢ LT.lt i k
          -/
          exact H.trans_lt hkj
          /-
            🎉 no goals
          -/
      · simp_rw [Fin.predAbove_of_le_castSucc i.castSucc _ (Fin.castSucc_le_castSucc_iff.mpr
        (Fin.succ_le_castSucc_iff.mpr h)), Fin.castPred_castSucc, ← Fin.succ_castSucc,
        Fin.succ_predAbove_succ]
        rw [Fin.predAbove_of_le_castSucc _ k.castSucc
        (Fin.castSucc_le_castSucc_iff.mpr (h.le.trans H)),
        Fin.castPred_castSucc, Fin.predAbove_of_le_castSucc _ k.succ
        (Fin.succ_le_castSucc_iff.mpr (H.trans_lt' h)), Fin.predAbove_of_le_castSucc _ k.succ
        (Fin.succ_le_castSucc_iff.mpr h)]


/--
If `f : [m] ⟶ [n+1]` is a morphism and `j` is not in the range of `f`,
then `factor_δ f j` is a morphism `[m] ⟶ [n]` such that
`factor_δ f j ≫ δ j = f` (as witnessed by `factor_δ_spec`).
-/
def factor_δ {m n : ℕ} (f : ([m] : SimplexCategory) ⟶ [n+1]) (j : Fin (n+2)) :
    ([m] : SimplexCategory) ⟶ [n] :=
  f ≫ σ (Fin.predAbove 0 j)


open Fin in
lemma factor_δ_spec {m n : ℕ} (f : ([m] : SimplexCategory) ⟶ [n+1]) (j : Fin (n+2))
    (hj : ∀ (k : Fin (m+1)), f.toOrderHom k ≠ j) :
    factor_δ f j ≫ δ j = f := by
  /-
    m n : Nat
    f : Quiver.Hom (SimplexCategory.mk m) (SimplexCategory.mk (HAdd.hAdd n 1))
    j : Fin (HAdd.hAdd n 2)
    hj : ∀ (k : Fin (HAdd.hAdd m 1)), Ne ((SimplexCategory.Hom.toOrderHom f) k) j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (SimplexCategory.factor_δ f j) (Simpl …
  -/
  ext k : 3
  /-
    case a.h.h
    m n : Nat
    f : Quiver.Hom (SimplexCategory.mk m) (SimplexCategory.mk (HAdd.hAdd n 1))
    j : Fin (HAdd.hAdd n 2)
    hj : ∀ (k : Fin (HAdd.hAdd m 1)), Ne ((SimplexCategory.Hom.toOrderHom f) k) j
    k : Fin (HAdd.hAdd (SimplexCategory.mk m).len 1)
    ⊢ Eq ((SimplexCategory.Hom.toOrderHom (CategoryTheory.CategoryStruct.comp (Sim …
  -/
  specialize hj k
  /-
    case a.h.h
    m n : Nat
    f : Quiver.Hom (SimplexCategory.mk m) (SimplexCategory.mk (HAdd.hAdd n 1))
    j : Fin (HAdd.hAdd n 2)
    k : Fin (HAdd.hAdd (SimplexCategory.mk m).len 1)
    hj : Ne ((SimplexCategory.Hom.toOrderHom f) k) j
    ⊢ Eq ((SimplexCategory.Hom.toOrderHom (CategoryTheory.CategoryStruct.comp (Sim …
  -/
  dsimp [factor_δ, δ, σ]
  /-
    case a.h.h
    m n : Nat
    f : Quiver.Hom (SimplexCategory.mk m) (SimplexCategory.mk (HAdd.hAdd n 1))
    j : Fin (HAdd.hAdd n 2)
    k : Fin (HAdd.hAdd (SimplexCategory.mk m).len 1)
    hj : Ne ((SimplexCategory.Hom.toOrderHom f) k) j
    ⊢ Eq (j.succAbove ((Fin.predAbove 0 j).predAbove ((SimplexCategory.Hom.toOrder …
  -/
  cases' j using cases with j
  · rw [predAbove_of_le_castSucc _ _ (zero_le _), castPred_zero, predAbove_of_castSucc_lt 0 _
    (castSucc_zero ▸ pos_of_ne_zero hj),
    zero_succAbove, succ_pred]
    /-
      case a.h.h.succ
      m n : Nat
      f : Quiver.Hom (SimplexCategory.mk m) (SimplexCategory.mk (HAdd.hAdd n 1))
      k : Fin (HAdd.hAdd (SimplexCategory.mk m).len 1)
      j : Fin (HAdd.hAdd n 1)
      hj : Ne ((SimplexCategory.Hom.toOrderHom f) k) j.succ
      ⊢ Eq (j.succ.succAbove ((Fin.predAbove 0 j.succ).predAbove ((SimplexCategory.H …
    -/
  · rw [predAbove_of_castSucc_lt 0 _ (castSucc_zero ▸ succ_pos _), pred_succ]
    /-
      case a.h.h.succ
      m n : Nat
      f : Quiver.Hom (SimplexCategory.mk m) (SimplexCategory.mk (HAdd.hAdd n 1))
      k : Fin (HAdd.hAdd (SimplexCategory.mk m).len 1)
      j : Fin (HAdd.hAdd n 1)
      hj : Ne ((SimplexCategory.Hom.toOrderHom f) k) j.succ
      ⊢ Eq (j.succ.succAbove (j.predAbove ((SimplexCategory.Hom.toOrderHom f) k))) ( …
    -/
    rcases hj.lt_or_lt with (hj | hj)
      /-
        case a.h.h.succ.inl
        m n : Nat
        f : Quiver.Hom (SimplexCategory.mk m) (SimplexCategory.mk (HAdd.hAdd n 1))
        k : Fin (HAdd.hAdd (SimplexCategory.mk m).len 1)
        j : Fin (HAdd.hAdd n 1)
        hj✝ : Ne ((SimplexCategory.Hom.toOrderHom f) k) j.succ
        hj : LT.lt ((SimplexCategory.Hom.toOrderHom f) k) j.succ
        ⊢ Eq (j.succ.succAbove (j.predAbove ((SimplexCategory.Hom.toOrderHom f) k))) ( …
      -/
    · rw [predAbove_of_le_castSucc j _]
      /-
        case a.h.h.succ.inl
        m n : Nat
        f : Quiver.Hom (SimplexCategory.mk m) (SimplexCategory.mk (HAdd.hAdd n 1))
        k : Fin (HAdd.hAdd (SimplexCategory.mk m).len 1)
        j : Fin (HAdd.hAdd n 1)
        hj✝ : Ne ((SimplexCategory.Hom.toOrderHom f) k) j.succ
        hj : LT.lt ((SimplexCategory.Hom.toOrderHom f) k) j.succ
        ⊢ Eq (j.succ.succAbove (((SimplexCategory.Hom.toOrderHom f) k).castPred ⋯)) (( …
      -/
      swap
        /-
          case a.h.h.succ.inl
          m n : Nat
          f : Quiver.Hom (SimplexCategory.mk m) (SimplexCategory.mk (HAdd.hAdd n 1))
          k : Fin (HAdd.hAdd (SimplexCategory.mk m).len 1)
          j : Fin (HAdd.hAdd n 1)
          hj✝ : Ne ((SimplexCategory.Hom.toOrderHom f) k) j.succ
          hj : LT.lt ((SimplexCategory.Hom.toOrderHom f) k) j.succ
          ⊢ LE.le ((SimplexCategory.Hom.toOrderHom f) k) j.castSucc
        -/
      · exact (le_castSucc_iff.mpr hj)
        /-
          🎉 no goals
        -/
        /-
          case a.h.h.succ.inl
          m n : Nat
          f : Quiver.Hom (SimplexCategory.mk m) (SimplexCategory.mk (HAdd.hAdd n 1))
          k : Fin (HAdd.hAdd (SimplexCategory.mk m).len 1)
          j : Fin (HAdd.hAdd n 1)
          hj✝ : Ne ((SimplexCategory.Hom.toOrderHom f) k) j.succ
          hj : LT.lt ((SimplexCategory.Hom.toOrderHom f) k) j.succ
          ⊢ Eq (j.succ.succAbove (((SimplexCategory.Hom.toOrderHom f) k).castPred ⋯)) (( …
        -/
      · rw [succAbove_of_castSucc_lt]
        /-
          case a.h.h.succ.inl
          m n : Nat
          f : Quiver.Hom (SimplexCategory.mk m) (SimplexCategory.mk (HAdd.hAdd n 1))
          k : Fin (HAdd.hAdd (SimplexCategory.mk m).len 1)
          j : Fin (HAdd.hAdd n 1)
          hj✝ : Ne ((SimplexCategory.Hom.toOrderHom f) k) j.succ
          hj : LT.lt ((SimplexCategory.Hom.toOrderHom f) k) j.succ
          ⊢ Eq (((SimplexCategory.Hom.toOrderHom f) k).castPred ⋯).castSucc ((SimplexCat …
        -/
        swap
          /-
            case a.h.h.succ.inl.h
            m n : Nat
            f : Quiver.Hom (SimplexCategory.mk m) (SimplexCategory.mk (HAdd.hAdd n 1))
            k : Fin (HAdd.hAdd (SimplexCategory.mk m).len 1)
            j : Fin (HAdd.hAdd n 1)
            hj✝ : Ne ((SimplexCategory.Hom.toOrderHom f) k) j.succ
            hj : LT.lt ((SimplexCategory.Hom.toOrderHom f) k) j.succ
            ⊢ LT.lt (((SimplexCategory.Hom.toOrderHom f) k).castPred ⋯).castSucc j.succ
          -/
        · rwa [castSucc_lt_succ_iff, castPred_le_iff, le_castSucc_iff]
          /-
            🎉 no goals
          -/
        /-
          case a.h.h.succ.inl
          m n : Nat
          f : Quiver.Hom (SimplexCategory.mk m) (SimplexCategory.mk (HAdd.hAdd n 1))
          k : Fin (HAdd.hAdd (SimplexCategory.mk m).len 1)
          j : Fin (HAdd.hAdd n 1)
          hj✝ : Ne ((SimplexCategory.Hom.toOrderHom f) k) j.succ
          hj : LT.lt ((SimplexCategory.Hom.toOrderHom f) k) j.succ
          ⊢ Eq (((SimplexCategory.Hom.toOrderHom f) k).castPred ⋯).castSucc ((SimplexCat …
        -/
        rw [castSucc_castPred]
        /-
          🎉 no goals
        -/
      /-
        case a.h.h.succ.inr
        m n : Nat
        f : Quiver.Hom (SimplexCategory.mk m) (SimplexCategory.mk (HAdd.hAdd n 1))
        k : Fin (HAdd.hAdd (SimplexCategory.mk m).len 1)
        j : Fin (HAdd.hAdd n 1)
        hj✝ : Ne ((SimplexCategory.Hom.toOrderHom f) k) j.succ
        hj : LT.lt j.succ ((SimplexCategory.Hom.toOrderHom f) k)
        ⊢ Eq (j.succ.succAbove (j.predAbove ((SimplexCategory.Hom.toOrderHom f) k))) ( …
      -/
    · rw [predAbove_of_castSucc_lt]
      /-
        case a.h.h.succ.inr
        m n : Nat
        f : Quiver.Hom (SimplexCategory.mk m) (SimplexCategory.mk (HAdd.hAdd n 1))
        k : Fin (HAdd.hAdd (SimplexCategory.mk m).len 1)
        j : Fin (HAdd.hAdd n 1)
        hj✝ : Ne ((SimplexCategory.Hom.toOrderHom f) k) j.succ
        hj : LT.lt j.succ ((SimplexCategory.Hom.toOrderHom f) k)
        ⊢ Eq (j.succ.succAbove (((SimplexCategory.Hom.toOrderHom f) k).pred ⋯)) ((Simp …
      -/
      swap
        /-
          case a.h.h.succ.inr.h
          m n : Nat
          f : Quiver.Hom (SimplexCategory.mk m) (SimplexCategory.mk (HAdd.hAdd n 1))
          k : Fin (HAdd.hAdd (SimplexCategory.mk m).len 1)
          j : Fin (HAdd.hAdd n 1)
          hj✝ : Ne ((SimplexCategory.Hom.toOrderHom f) k) j.succ
          hj : LT.lt j.succ ((SimplexCategory.Hom.toOrderHom f) k)
          ⊢ LT.lt j.castSucc ((SimplexCategory.Hom.toOrderHom f) k)
        -/
      · exact (castSucc_lt_succ _).trans hj
        /-
          🎉 no goals
        -/
      /-
        case a.h.h.succ.inr
        m n : Nat
        f : Quiver.Hom (SimplexCategory.mk m) (SimplexCategory.mk (HAdd.hAdd n 1))
        k : Fin (HAdd.hAdd (SimplexCategory.mk m).len 1)
        j : Fin (HAdd.hAdd n 1)
        hj✝ : Ne ((SimplexCategory.Hom.toOrderHom f) k) j.succ
        hj : LT.lt j.succ ((SimplexCategory.Hom.toOrderHom f) k)
        ⊢ Eq (j.succ.succAbove (((SimplexCategory.Hom.toOrderHom f) k).pred ⋯)) ((Simp …
      -/
      rw [succAbove_of_le_castSucc]
      /-
        case a.h.h.succ.inr
        m n : Nat
        f : Quiver.Hom (SimplexCategory.mk m) (SimplexCategory.mk (HAdd.hAdd n 1))
        k : Fin (HAdd.hAdd (SimplexCategory.mk m).len 1)
        j : Fin (HAdd.hAdd n 1)
        hj✝ : Ne ((SimplexCategory.Hom.toOrderHom f) k) j.succ
        hj : LT.lt j.succ ((SimplexCategory.Hom.toOrderHom f) k)
        ⊢ Eq (((SimplexCategory.Hom.toOrderHom f) k).pred ⋯).succ ((SimplexCategory.Ho …
      -/
      swap
        /-
          case a.h.h.succ.inr.h
          m n : Nat
          f : Quiver.Hom (SimplexCategory.mk m) (SimplexCategory.mk (HAdd.hAdd n 1))
          k : Fin (HAdd.hAdd (SimplexCategory.mk m).len 1)
          j : Fin (HAdd.hAdd n 1)
          hj✝ : Ne ((SimplexCategory.Hom.toOrderHom f) k) j.succ
          hj : LT.lt j.succ ((SimplexCategory.Hom.toOrderHom f) k)
          ⊢ LE.le j.succ (((SimplexCategory.Hom.toOrderHom f) k).pred ⋯).castSucc
        -/
      · rwa [succ_le_castSucc_iff, lt_pred_iff]
        /-
          🎉 no goals
        -/
      /-
        case a.h.h.succ.inr
        m n : Nat
        f : Quiver.Hom (SimplexCategory.mk m) (SimplexCategory.mk (HAdd.hAdd n 1))
        k : Fin (HAdd.hAdd (SimplexCategory.mk m).len 1)
        j : Fin (HAdd.hAdd n 1)
        hj✝ : Ne ((SimplexCategory.Hom.toOrderHom f) k) j.succ
        hj : LT.lt j.succ ((SimplexCategory.Hom.toOrderHom f) k)
        ⊢ Eq (((SimplexCategory.Hom.toOrderHom f) k).pred ⋯).succ ((SimplexCategory.Ho …
      -/
      rw [succ_pred]
      /-
        🎉 no goals
      -/


@[simp]
lemma δ_zero_mkOfSucc {n : ℕ} (i : Fin n) :
    δ 0 ≫ mkOfSucc i = SimplexCategory.const _ [n] i.succ := by
  /-
    n : Nat
    i : Fin n
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (SimplexCategory.δ 0) (SimplexCategor …
  -/
  ext x
  /-
    case a.h.h.h
    n : Nat
    i : Fin n
    x : Fin (HAdd.hAdd (SimplexCategory.mk 0).len 1)
    ⊢ Eq ↑((SimplexCategory.Hom.toOrderHom (CategoryTheory.CategoryStruct.comp (Si …
  -/
  fin_cases x
  /-
    case a.h.h.h.«_@»._hyg.936.«0»
    n : Nat
    i : Fin n
    ⊢ Eq ↑((SimplexCategory.Hom.toOrderHom (CategoryTheory.CategoryStruct.comp (Si …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
lemma δ_one_mkOfSucc {n : ℕ} (i : Fin n) :
    δ 1 ≫ mkOfSucc i = SimplexCategory.const _ _ i.castSucc := by
  /-
    n : Nat
    i : Fin n
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (SimplexCategory.δ 1) (SimplexCategor …
  -/
  ext x
  /-
    case a.h.h.h
    n : Nat
    i : Fin n
    x : Fin (HAdd.hAdd (SimplexCategory.mk 0).len 1)
    ⊢ Eq ↑((SimplexCategory.Hom.toOrderHom (CategoryTheory.CategoryStruct.comp (Si …
  -/
  fin_cases x
  /-
    case a.h.h.h.«_@»._hyg.936.«0»
    n : Nat
    i : Fin n
    ⊢ Eq ↑((SimplexCategory.Hom.toOrderHom (CategoryTheory.CategoryStruct.comp (Si …
  -/
  aesop
  /-
    🎉 no goals
  -/


/-- If `i + 1 < j`, `mkOfSucc i ≫ δ j` is the morphism `[1] ⟶ [n]` that
sends `0` and `1` to `i` and `i + 1`, respectively. -/
lemma mkOfSucc_δ_lt {n : ℕ} {i : Fin n} {j : Fin (n + 2)}
    (h : i.succ.castSucc < j) :
    mkOfSucc i ≫ δ j = mkOfSucc i.castSucc := by
  /-
    n : Nat
    i : Fin n
    j : Fin (HAdd.hAdd n 2)
    h : LT.lt i.succ.castSucc j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (SimplexCategory.mkOfSucc i) (Simplex …
  -/
  ext x
  /-
    case a.h.h.h
    n : Nat
    i : Fin n
    j : Fin (HAdd.hAdd n 2)
    h : LT.lt i.succ.castSucc j
    x : Fin (HAdd.hAdd (SimplexCategory.mk 1).len 1)
    ⊢ Eq ↑((SimplexCategory.Hom.toOrderHom (CategoryTheory.CategoryStruct.comp (Si …
  -/
  fin_cases x
    /-
      case a.h.h.h.«_@»._hyg.936.«0»
      n : Nat
      i : Fin n
      j : Fin (HAdd.hAdd n 2)
      h : LT.lt i.succ.castSucc j
      ⊢ Eq ↑((SimplexCategory.Hom.toOrderHom (CategoryTheory.CategoryStruct.comp (Si …
    -/
  · simp [δ, Fin.succAbove_of_castSucc_lt _ _ (Nat.lt_trans _ h)]
    /-
      🎉 no goals
    -/
    /-
      case a.h.h.h.«_@»._hyg.936.«1»
      n : Nat
      i : Fin n
      j : Fin (HAdd.hAdd n 2)
      h : LT.lt i.succ.castSucc j
      ⊢ Eq ↑((SimplexCategory.Hom.toOrderHom (CategoryTheory.CategoryStruct.comp (Si …
    -/
  · simp [δ, Fin.succAbove_of_castSucc_lt _ _ h]
    /-
      🎉 no goals
    -/


/-- If `i + 1 > j`, `mkOfSucc i ≫ δ j` is the morphism `[1] ⟶ [n]` that
sends `0` and `1` to `i + 1` and `i + 2`, respectively. -/
lemma mkOfSucc_δ_gt {n : ℕ} {i : Fin n} {j : Fin (n + 2)}
    (h : j < i.succ.castSucc) :
    mkOfSucc i ≫ δ j = mkOfSucc i.succ := by
  /-
    n : Nat
    i : Fin n
    j : Fin (HAdd.hAdd n 2)
    h : LT.lt j i.succ.castSucc
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (SimplexCategory.mkOfSucc i) (Simplex …
  -/
  ext x
  simp only [δ, len_mk, mkHom, comp_toOrderHom, Hom.toOrderHom_mk, OrderHom.comp_coe,
    OrderEmbedding.toOrderHom_coe, Function.comp_apply, Fin.succAboveOrderEmb_apply]
  /-
    case a.h.h.h
    n : Nat
    i : Fin n
    j : Fin (HAdd.hAdd n 2)
    h : LT.lt j i.succ.castSucc
    x : Fin (HAdd.hAdd (SimplexCategory.mk 1).len 1)
    ⊢ Eq ↑(j.succAbove ((SimplexCategory.Hom.toOrderHom (SimplexCategory.mkOfSucc  …
  -/
  fin_cases x <;> rw [Fin.succAbove_of_le_castSucc]
    /-
      case a.h.h.h.«_@»._hyg.936.«0»
      n : Nat
      i : Fin n
      j : Fin (HAdd.hAdd n 2)
      h : LT.lt j i.succ.castSucc
      ⊢ Eq ↑((SimplexCategory.Hom.toOrderHom (SimplexCategory.mkOfSucc i)) ((fun i = …
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case a.h.h.h.«_@»._hyg.936.«0».h
      n : Nat
      i : Fin n
      j : Fin (HAdd.hAdd n 2)
      h : LT.lt j i.succ.castSucc
      ⊢ LE.le j ((SimplexCategory.Hom.toOrderHom (SimplexCategory.mkOfSucc i)) ((fun …
    -/
  · exact Nat.le_of_lt_succ h
    /-
      🎉 no goals
    -/
    /-
      case a.h.h.h.«_@»._hyg.936.«1»
      n : Nat
      i : Fin n
      j : Fin (HAdd.hAdd n 2)
      h : LT.lt j i.succ.castSucc
      ⊢ Eq ↑((SimplexCategory.Hom.toOrderHom (SimplexCategory.mkOfSucc i)) ((fun i = …
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case a.h.h.h.«_@»._hyg.936.«1».h
      n : Nat
      i : Fin n
      j : Fin (HAdd.hAdd n 2)
      h : LT.lt j i.succ.castSucc
      ⊢ LE.le j ((SimplexCategory.Hom.toOrderHom (SimplexCategory.mkOfSucc i)) ((fun …
    -/
  · exact Nat.le_of_lt h
    /-
      🎉 no goals
    -/


/-- If `i + 1 = j`, `mkOfSucc i ≫ δ j` is the morphism `[1] ⟶ [n]` that
sends `0` and `1` to `i` and `i + 2`, respectively. -/
lemma mkOfSucc_δ_eq {n : ℕ} {i : Fin n} {j : Fin (n + 2)}
    (h : j = i.succ.castSucc) :
                                            /-
                                              n : Nat
                                              i : Fin n
                                              j : Fin (HAdd.hAdd n 2)
                                              h : Eq j i.succ.castSucc
                                              ⊢ LE.le (HAdd.hAdd (↑i) 2) (HAdd.hAdd n 1)
                                            -/
    mkOfSucc i ≫ δ j = intervalEdge i 2 (by omega) := by
                                            /-
                                              🎉 no goals
                                            -/
  /-
    n : Nat
    i : Fin n
    j : Fin (HAdd.hAdd n 2)
    h : Eq j i.succ.castSucc
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (SimplexCategory.mkOfSucc i) (Simplex …
  -/
  ext x
  /-
    case a.h.h.h
    n : Nat
    i : Fin n
    j : Fin (HAdd.hAdd n 2)
    h : Eq j i.succ.castSucc
    x : Fin (HAdd.hAdd (SimplexCategory.mk 1).len 1)
    ⊢ Eq ↑((SimplexCategory.Hom.toOrderHom (CategoryTheory.CategoryStruct.comp (Si …
  -/
  fin_cases x
    /-
      case a.h.h.h.«_@»._hyg.936.«0»
      n : Nat
      i : Fin n
      j : Fin (HAdd.hAdd n 2)
      h : Eq j i.succ.castSucc
      ⊢ Eq ↑((SimplexCategory.Hom.toOrderHom (CategoryTheory.CategoryStruct.comp (Si …
    -/
  · subst h
    simp only [δ, len_mk, Nat.reduceAdd, mkHom, comp_toOrderHom, Hom.toOrderHom_mk,
      Fin.zero_eta, OrderHom.comp_coe, OrderEmbedding.toOrderHom_coe, Function.comp_apply,
      mkOfSucc_homToOrderHom_zero, Fin.succAboveOrderEmb_apply,
      Fin.castSucc_succAbove_castSucc, Fin.succAbove_succ_self]
    /-
      case a.h.h.h.«_@»._hyg.936.«0»
      n : Nat
      i : Fin n
      ⊢ Eq ↑i.castSucc.castSucc ↑((SimplexCategory.Hom.toOrderHom (SimplexCategory.i …
    -/
    rfl
    /-
      🎉 no goals
    -/
  · simp only [δ, len_mk, Nat.reduceAdd, mkHom, comp_toOrderHom, Hom.toOrderHom_mk, Fin.mk_one,
      OrderHom.comp_coe, OrderEmbedding.toOrderHom_coe, Function.comp_apply,
      mkOfSucc_homToOrderHom_one, Fin.succAboveOrderEmb_apply]
    /-
      case a.h.h.h.«_@»._hyg.936.«1»
      n : Nat
      i : Fin n
      j : Fin (HAdd.hAdd n 2)
      h : Eq j i.succ.castSucc
      ⊢ Eq ↑(j.succAbove i.succ) ↑((SimplexCategory.Hom.toOrderHom (SimplexCategory. …
    -/
    subst h
    /-
      case a.h.h.h.«_@»._hyg.936.«1»
      n : Nat
      i : Fin n
      ⊢ Eq ↑(i.succ.castSucc.succAbove i.succ) ↑((SimplexCategory.Hom.toOrderHom (Si …
    -/
    rw [Fin.succAbove_castSucc_self]
    /-
      case a.h.h.h.«_@»._hyg.936.«1»
      n : Nat
      i : Fin n
      ⊢ Eq ↑i.succ.succ ↑((SimplexCategory.Hom.toOrderHom (SimplexCategory.intervalE …
    -/
    rfl
    /-
      🎉 no goals
    -/


theorem eq_of_one_to_two (f : ([1] : SimplexCategory) ⟶ [2]) :
    f = (δ (n := 1) 0) ∨ f = (δ (n := 1) 1) ∨ f = (δ (n := 1) 2) ∨
      ∃ a, f = SimplexCategory.const _ _ a := by
  /-
    f : Quiver.Hom (SimplexCategory.mk 1) (SimplexCategory.mk 2)
    ⊢ Or (Eq f (SimplexCategory.δ 0)) (Or (Eq f (SimplexCategory.δ 1)) (Or (Eq f ( …
  -/
  have : f.toOrderHom 0 ≤ f.toOrderHom 1 := f.toOrderHom.monotone (by decide : (0 : Fin 2) ≤ 1)
  match e0 : f.toOrderHom 0, e1 : f.toOrderHom 1 with
  | 1, 2 =>
    left
    ext i : 3
    match i with
    | 0 => exact e0
    | 1 => exact e1
  | 0, 2 =>
    right; left
    ext i : 3
    match i with
    | 0 => exact e0
    | 1 => exact e1
  | 0, 1 =>
    right; right; left
    ext i : 3
    match i with
    | 0 => exact e0
    | 1 => exact e1
  | 0, 0 | 1, 1 | 2, 2 =>
    right; right; right; use f.toOrderHom 0
    ext i : 3
    match i with
    | 0 => rfl
    | 1 => exact e1.trans e0.symm
  | 1, 0 | 2, 0 | 2, 1 =>
    rw [e0, e1] at this
    exact Not.elim (by decide) this


/-- The functor that exhibits `SimplexCategory` as skeleton
of `NonemptyFinLinOrd` -/
@[simps obj map]
def skeletalFunctor : SimplexCategory ⥤ NonemptyFinLinOrd where
  obj a := NonemptyFinLinOrd.of (Fin (a.len + 1))
  map f := f.toOrderHom


theorem skeletalFunctor.coe_map {Δ₁ Δ₂ : SimplexCategory} (f : Δ₁ ⟶ Δ₂) :
    ↑(skeletalFunctor.map f) = f.toOrderHom :=
  rfl


theorem skeletal : Skeletal SimplexCategory := fun X Y ⟨I⟩ => by
  suffices Fintype.card (Fin (X.len + 1)) = Fintype.card (Fin (Y.len + 1)) by
    ext
    simpa
  /-
    X Y : SimplexCategory
    x✝ : CategoryTheory.IsIsomorphic X Y
    I : CategoryTheory.Iso X Y
    ⊢ Eq (Fintype.card (Fin (HAdd.hAdd X.len 1))) (Fintype.card (Fin (HAdd.hAdd Y. …
  -/
  apply Fintype.card_congr
  /-
    case f
    X Y : SimplexCategory
    x✝ : CategoryTheory.IsIsomorphic X Y
    I : CategoryTheory.Iso X Y
    ⊢ Equiv (Fin (HAdd.hAdd X.len 1)) (Fin (HAdd.hAdd Y.len 1))
  -/
  exact ((skeletalFunctor ⋙ forget NonemptyFinLinOrd).mapIso I).toEquiv
  /-
    🎉 no goals
  -/


instance : skeletalFunctor.Full where
  map_surjective f := ⟨SimplexCategory.Hom.mk f, rfl⟩


instance : skeletalFunctor.Faithful where
  map_injective {_ _ f g} h := by
    /-
      x✝¹ x✝ : SimplexCategory
      f g : Quiver.Hom x✝¹ x✝
      h : Eq (SimplexCategory.skeletalFunctor.map f) (SimplexCategory.skeletalFuncto …
      ⊢ Eq f g
    -/
    ext1
    /-
      case a
      x✝¹ x✝ : SimplexCategory
      f g : Quiver.Hom x✝¹ x✝
      h : Eq (SimplexCategory.skeletalFunctor.map f) (SimplexCategory.skeletalFuncto …
      ⊢ Eq (SimplexCategory.Hom.toOrderHom f) (SimplexCategory.Hom.toOrderHom g)
    -/
    exact h
    /-
      🎉 no goals
    -/


instance : skeletalFunctor.EssSurj where
  mem_essImage X :=
    ⟨mk (Fintype.card X - 1 : ℕ),
      ⟨by
        have aux : Fintype.card X = Fintype.card X - 1 + 1 :=
          (Nat.succ_pred_eq_of_pos <| Fintype.card_pos_iff.mpr ⟨⊥⟩).symm
        /-
          X : NonemptyFinLinOrd
          aux : Eq (Fintype.card ↑X) (HAdd.hAdd (HSub.hSub (Fintype.card ↑X) 1) 1)
          ⊢ CategoryTheory.Iso (SimplexCategory.skeletalFunctor.obj (SimplexCategory.mk  …
        -/
        let f := monoEquivOfFin X aux
        /-
          X : NonemptyFinLinOrd
          aux : Eq (Fintype.card ↑X) (HAdd.hAdd (HSub.hSub (Fintype.card ↑X) 1) 1)
          f : OrderIso (Fin (HAdd.hAdd (HSub.hSub (Fintype.card ↑X) 1) 1)) ↑X := monoEqu …
          ⊢ CategoryTheory.Iso (SimplexCategory.skeletalFunctor.obj (SimplexCategory.mk  …
        -/
        have hf := (Finset.univ.orderEmbOfFin aux).strictMono
        refine
          { hom := ⟨f, hf.monotone⟩
            inv := ⟨f.symm, ?_⟩
            hom_inv_id := by ext1; apply f.symm_apply_apply
            inv_hom_id := by ext1; apply f.apply_symm_apply }
        /-
          X : NonemptyFinLinOrd
          aux : Eq (Fintype.card ↑X) (HAdd.hAdd (HSub.hSub (Fintype.card ↑X) 1) 1)
          f : OrderIso (Fin (HAdd.hAdd (HSub.hSub (Fintype.card ↑X) 1) 1)) ↑X := monoEqu …
          hf : StrictMono ⇑(Finset.univ.orderEmbOfFin aux)
          ⊢ Monotone ⇑f.symm
        -/
        intro i j h
        /-
          X : NonemptyFinLinOrd
          aux : Eq (Fintype.card ↑X) (HAdd.hAdd (HSub.hSub (Fintype.card ↑X) 1) 1)
          f : OrderIso (Fin (HAdd.hAdd (HSub.hSub (Fintype.card ↑X) 1) 1)) ↑X := monoEqu …
          hf : StrictMono ⇑(Finset.univ.orderEmbOfFin aux)
          i j : ↑X
          h : LE.le i j
          ⊢ LE.le (f.symm i) (f.symm j)
        -/
        show f.symm i ≤ f.symm j
        /-
          X : NonemptyFinLinOrd
          aux : Eq (Fintype.card ↑X) (HAdd.hAdd (HSub.hSub (Fintype.card ↑X) 1) 1)
          f : OrderIso (Fin (HAdd.hAdd (HSub.hSub (Fintype.card ↑X) 1) 1)) ↑X := monoEqu …
          hf : StrictMono ⇑(Finset.univ.orderEmbOfFin aux)
          i j : ↑X
          h : LE.le i j
          ⊢ LE.le (f.symm i) (f.symm j)
        -/
        rw [← hf.le_iff_le]
        /-
          X : NonemptyFinLinOrd
          aux : Eq (Fintype.card ↑X) (HAdd.hAdd (HSub.hSub (Fintype.card ↑X) 1) 1)
          f : OrderIso (Fin (HAdd.hAdd (HSub.hSub (Fintype.card ↑X) 1) 1)) ↑X := monoEqu …
          hf : StrictMono ⇑(Finset.univ.orderEmbOfFin aux)
          i j : ↑X
          h : LE.le i j
          ⊢ LE.le ((Finset.univ.orderEmbOfFin aux) (f.symm i)) ((Finset.univ.orderEmbOfF …
        -/
        show f (f.symm i) ≤ f (f.symm j)
        /-
          X : NonemptyFinLinOrd
          aux : Eq (Fintype.card ↑X) (HAdd.hAdd (HSub.hSub (Fintype.card ↑X) 1) 1)
          f : OrderIso (Fin (HAdd.hAdd (HSub.hSub (Fintype.card ↑X) 1) 1)) ↑X := monoEqu …
          hf : StrictMono ⇑(Finset.univ.orderEmbOfFin aux)
          i j : ↑X
          h : LE.le i j
          ⊢ LE.le (f (f.symm i)) (f (f.symm j))
        -/
        simpa only [OrderIso.apply_symm_apply]⟩⟩
        /-
          🎉 no goals
        -/


noncomputable instance isEquivalence : skeletalFunctor.IsEquivalence where


/-- The equivalence that exhibits `SimplexCategory` as skeleton
of `NonemptyFinLinOrd` -/
noncomputable def skeletalEquivalence : SimplexCategory ≌ NonemptyFinLinOrd :=
  Functor.asEquivalence skeletalFunctor


/-- `SimplexCategory` is a skeleton of `NonemptyFinLinOrd`.
-/
lemma isSkeletonOf :
    IsSkeletonOf NonemptyFinLinOrd SimplexCategory skeletalFunctor where
  skel := skeletal
  eqv := SkeletalFunctor.isEquivalence


/-- The truncated simplex category. -/
def Truncated (n : ℕ) :=
  FullSubcategory fun a : SimplexCategory => a.len ≤ n


instance (n : ℕ) : SmallCategory.{0} (Truncated n) :=
  FullSubcategory.category _


instance {n} : Inhabited (Truncated n) :=
            /-
              n : Nat
              ⊢ LE.le (SimplexCategory.mk 0).len n
            -/
  ⟨⟨[0], by simp⟩⟩
            /-
              🎉 no goals
            -/


/-- The fully faithful inclusion of the truncated simplex category into the usual
simplex category.
-/
def inclusion (n : ℕ) : SimplexCategory.Truncated n ⥤ SimplexCategory :=
  fullSubcategoryInclusion _


instance (n : ℕ) : (inclusion n : Truncated n ⥤ _).Full := FullSubcategory.full _

instance (n : ℕ) : (inclusion n : Truncated n ⥤ _).Faithful := FullSubcategory.faithful _


/-- A proof that the full subcategory inclusion is fully faithful.-/
noncomputable def inclusion.fullyFaithful (n : ℕ) :
    (inclusion n : Truncated n ⥤ _).op.FullyFaithful := Functor.FullyFaithful.ofFullyFaithful _


@[ext]
theorem Hom.ext {n} {a b : Truncated n} (f g : a ⟶ b) :
    f.toOrderHom = g.toOrderHom → f = g := SimplexCategory.Hom.ext _ _


instance : ConcreteCategory.{0} SimplexCategory where
  forget :=
    { obj := fun i => Fin (i.len + 1)
      map := fun f => f.toOrderHom }
                                  /-
                                    X✝ Y✝ : SimplexCategory
                                    a₁✝ a₂✝ : Quiver.Hom X✝ Y✝
                                    h : Eq ({ obj := fun i => Fin (HAdd.hAdd i.len 1), map := fun {X Y} f => ⇑(Sim …
                                    ⊢ Eq a₁✝ a₂✝
                                  -/
  forget_faithful := ⟨fun h => by ext : 2; exact h⟩
                                           /-
                                             🎉 no goals
                                           -/


/-- A morphism in `SimplexCategory` is a monomorphism precisely when it is an injective function
-/
theorem mono_iff_injective {n m : SimplexCategory} {f : n ⟶ m} :
    Mono f ↔ Function.Injective f.toOrderHom := by
  /-
    n m : SimplexCategory
    f : Quiver.Hom n m
    ⊢ Iff (CategoryTheory.Mono f) (Function.Injective ⇑(SimplexCategory.Hom.toOrde …
  -/
  rw [← Functor.mono_map_iff_mono skeletalEquivalence.functor]
  /-
    n m : SimplexCategory
    f : Quiver.Hom n m
    ⊢ Iff (CategoryTheory.Mono (SimplexCategory.skeletalEquivalence.functor.map f) …
  -/
  dsimp only [skeletalEquivalence, Functor.asEquivalence_functor]
  simp only [skeletalFunctor_obj, skeletalFunctor_map,
    NonemptyFinLinOrd.mono_iff_injective, NonemptyFinLinOrd.coe_of]


/-- A morphism in `SimplexCategory` is an epimorphism if and only if it is a surjective function
-/
theorem epi_iff_surjective {n m : SimplexCategory} {f : n ⟶ m} :
    Epi f ↔ Function.Surjective f.toOrderHom := by
  /-
    n m : SimplexCategory
    f : Quiver.Hom n m
    ⊢ Iff (CategoryTheory.Epi f) (Function.Surjective ⇑(SimplexCategory.Hom.toOrde …
  -/
  rw [← Functor.epi_map_iff_epi skeletalEquivalence.functor]
  /-
    n m : SimplexCategory
    f : Quiver.Hom n m
    ⊢ Iff (CategoryTheory.Epi (SimplexCategory.skeletalEquivalence.functor.map f)) …
  -/
  dsimp only [skeletalEquivalence, Functor.asEquivalence_functor]
  simp only [skeletalFunctor_obj, skeletalFunctor_map,
    NonemptyFinLinOrd.epi_iff_surjective, NonemptyFinLinOrd.coe_of]


/-- A monomorphism in `SimplexCategory` must increase lengths -/
theorem len_le_of_mono {x y : SimplexCategory} {f : x ⟶ y} : Mono f → x.len ≤ y.len := by
  /-
    x y : SimplexCategory
    f : Quiver.Hom x y
    ⊢ CategoryTheory.Mono f → LE.le x.len y.len
  -/
  intro hyp_f_mono
  /-
    x y : SimplexCategory
    f : Quiver.Hom x y
    hyp_f_mono : CategoryTheory.Mono f
    ⊢ LE.le x.len y.len
  -/
  have f_inj : Function.Injective f.toOrderHom.toFun := mono_iff_injective.1 hyp_f_mono
  /-
    x y : SimplexCategory
    f : Quiver.Hom x y
    hyp_f_mono : CategoryTheory.Mono f
    f_inj : Function.Injective (SimplexCategory.Hom.toOrderHom f).toFun
    ⊢ LE.le x.len y.len
  -/
  simpa using Fintype.card_le_of_injective f.toOrderHom.toFun f_inj
  /-
    🎉 no goals
  -/


theorem le_of_mono {n m : ℕ} {f : ([n] : SimplexCategory) ⟶ [m]} : CategoryTheory.Mono f → n ≤ m :=
  len_le_of_mono


/-- An epimorphism in `SimplexCategory` must decrease lengths -/
theorem len_le_of_epi {x y : SimplexCategory} {f : x ⟶ y} : Epi f → y.len ≤ x.len := by
  /-
    x y : SimplexCategory
    f : Quiver.Hom x y
    ⊢ CategoryTheory.Epi f → LE.le y.len x.len
  -/
  intro hyp_f_epi
  /-
    x y : SimplexCategory
    f : Quiver.Hom x y
    hyp_f_epi : CategoryTheory.Epi f
    ⊢ LE.le y.len x.len
  -/
  have f_surj : Function.Surjective f.toOrderHom.toFun := epi_iff_surjective.1 hyp_f_epi
  /-
    x y : SimplexCategory
    f : Quiver.Hom x y
    hyp_f_epi : CategoryTheory.Epi f
    f_surj : Function.Surjective (SimplexCategory.Hom.toOrderHom f).toFun
    ⊢ LE.le y.len x.len
  -/
  simpa using Fintype.card_le_of_surjective f.toOrderHom.toFun f_surj
  /-
    🎉 no goals
  -/


theorem le_of_epi {n m : ℕ} {f : ([n] : SimplexCategory) ⟶ [m]} : Epi f → m ≤ n :=
  len_le_of_epi


instance {n : ℕ} {i : Fin (n + 2)} : Mono (δ i) := by
  /-
    n : Nat
    i : Fin (HAdd.hAdd n 2)
    ⊢ CategoryTheory.Mono (SimplexCategory.δ i)
  -/
  rw [mono_iff_injective]
  /-
    n : Nat
    i : Fin (HAdd.hAdd n 2)
    ⊢ Function.Injective ⇑(SimplexCategory.Hom.toOrderHom (SimplexCategory.δ i))
  -/
  exact Fin.succAbove_right_injective
  /-
    🎉 no goals
  -/


instance {n : ℕ} {i : Fin (n + 1)} : Epi (σ i) := by
  /-
    n : Nat
    i : Fin (HAdd.hAdd n 1)
    ⊢ CategoryTheory.Epi (SimplexCategory.σ i)
  -/
  rw [epi_iff_surjective]
  /-
    n : Nat
    i : Fin (HAdd.hAdd n 1)
    ⊢ Function.Surjective ⇑(SimplexCategory.Hom.toOrderHom (SimplexCategory.σ i))
  -/
  intro b
  /-
    n : Nat
    i : Fin (HAdd.hAdd n 1)
    b : Fin (HAdd.hAdd (SimplexCategory.mk n).len 1)
    ⊢ Exists fun a => Eq ((SimplexCategory.Hom.toOrderHom (SimplexCategory.σ i)) a …
  -/
  simp only [σ, mkHom, Hom.toOrderHom_mk, OrderHom.coe_mk]
  /-
    n : Nat
    i : Fin (HAdd.hAdd n 1)
    b : Fin (HAdd.hAdd (SimplexCategory.mk n).len 1)
    ⊢ Exists fun a => Eq ({ toFun := i.predAbove, monotone' := ⋯ } a) b
  -/
  by_cases h : b ≤ i
    /-
      case pos
      n : Nat
      i : Fin (HAdd.hAdd n 1)
      b : Fin (HAdd.hAdd (SimplexCategory.mk n).len 1)
      h : LE.le b i
      ⊢ Exists fun a => Eq ({ toFun := i.predAbove, monotone' := ⋯ } a) b
    -/
  · use b
    -- This was not needed before https://github.com/leanprover/lean4/pull/2644
    /-
      case h
      n : Nat
      i : Fin (HAdd.hAdd n 1)
      b : Fin (HAdd.hAdd (SimplexCategory.mk n).len 1)
      h : LE.le b i
      ⊢ Eq ({ toFun := i.predAbove, monotone' := ⋯ } ↑↑b) b
    -/
    dsimp
    /-
      case h
      n : Nat
      i : Fin (HAdd.hAdd n 1)
      b : Fin (HAdd.hAdd (SimplexCategory.mk n).len 1)
      h : LE.le b i
      ⊢ Eq (i.predAbove ↑↑b) b
    -/
    rw [Fin.predAbove_of_le_castSucc i b (by simpa only [Fin.coe_eq_castSucc] using h)]
    /-
      case h
      n : Nat
      i : Fin (HAdd.hAdd n 1)
      b : Fin (HAdd.hAdd (SimplexCategory.mk n).len 1)
      h : LE.le b i
      ⊢ Eq ((↑↑b).castPred ⋯) b
    -/
    simp only [len_mk, Fin.coe_eq_castSucc, Fin.castPred_castSucc]
    /-
      🎉 no goals
    -/
    /-
      case neg
      n : Nat
      i : Fin (HAdd.hAdd n 1)
      b : Fin (HAdd.hAdd (SimplexCategory.mk n).len 1)
      h : Not (LE.le b i)
      ⊢ Exists fun a => Eq ({ toFun := i.predAbove, monotone' := ⋯ } a) b
    -/
  · use b.succ
    -- This was not needed before https://github.com/leanprover/lean4/pull/2644
    /-
      case h
      n : Nat
      i : Fin (HAdd.hAdd n 1)
      b : Fin (HAdd.hAdd (SimplexCategory.mk n).len 1)
      h : Not (LE.le b i)
      ⊢ Eq ({ toFun := i.predAbove, monotone' := ⋯ } b.succ) b
    -/
    dsimp
    /-
      case h
      n : Nat
      i : Fin (HAdd.hAdd n 1)
      b : Fin (HAdd.hAdd (SimplexCategory.mk n).len 1)
      h : Not (LE.le b i)
      ⊢ Eq (i.predAbove b.succ) b
    -/
    rw [Fin.predAbove_of_castSucc_lt i b.succ _, Fin.pred_succ]
    /-
      n : Nat
      i : Fin (HAdd.hAdd n 1)
      b : Fin (HAdd.hAdd (SimplexCategory.mk n).len 1)
      h : Not (LE.le b i)
      ⊢ LT.lt i.castSucc b.succ
    -/
    rw [not_le] at h
    /-
      n : Nat
      i : Fin (HAdd.hAdd n 1)
      b : Fin (HAdd.hAdd (SimplexCategory.mk n).len 1)
      h : LT.lt i b
      ⊢ LT.lt i.castSucc b.succ
    -/
    rw [Fin.lt_iff_val_lt_val] at h ⊢
    /-
      n : Nat
      i : Fin (HAdd.hAdd n 1)
      b : Fin (HAdd.hAdd (SimplexCategory.mk n).len 1)
      h : LT.lt ↑i ↑b
      ⊢ LT.lt ↑i.castSucc ↑b.succ
    -/
    simpa only [Fin.val_succ, Fin.coe_castSucc] using Nat.lt.step h
    /-
      🎉 no goals
    -/


instance : (forget SimplexCategory).ReflectsIsomorphisms :=
  ⟨fun f hf =>
    Iso.isIso_hom
      { hom := f
        inv := Hom.mk
            { toFun := inv ((forget SimplexCategory).map f)
              monotone' := fun y₁ y₂ h => by
                /-
                  A✝ B✝ : SimplexCategory
                  f : Quiver.Hom A✝ B✝
                  hf : CategoryTheory.IsIso ((CategoryTheory.forget SimplexCategory).map f)
                  y₁ y₂ : Fin (HAdd.hAdd B✝.len 1)
                  h : LE.le y₁ y₂
                  ⊢ LE.le (CategoryTheory.inv ((CategoryTheory.forget SimplexCategory).map f) y₁ …
                -/
                by_cases h' : y₁ < y₂
                  /-
                    case pos
                    A✝ B✝ : SimplexCategory
                    f : Quiver.Hom A✝ B✝
                    hf : CategoryTheory.IsIso ((CategoryTheory.forget SimplexCategory).map f)
                    y₁ y₂ : Fin (HAdd.hAdd B✝.len 1)
                    h : LE.le y₁ y₂
                    h' : LT.lt y₁ y₂
                    ⊢ LE.le (CategoryTheory.inv ((CategoryTheory.forget SimplexCategory).map f) y₁ …
                  -/
                · by_contra h''
                  /-
                    case pos
                    A✝ B✝ : SimplexCategory
                    f : Quiver.Hom A✝ B✝
                    hf : CategoryTheory.IsIso ((CategoryTheory.forget SimplexCategory).map f)
                    y₁ y₂ : Fin (HAdd.hAdd B✝.len 1)
                    h : LE.le y₁ y₂
                    h' : LT.lt y₁ y₂
                    h'' : Not (LE.le (CategoryTheory.inv ((CategoryTheory.forget SimplexCategory). …
                    ⊢ False
                  -/
                  apply not_le.mpr h'
                  /-
                    case pos
                    A✝ B✝ : SimplexCategory
                    f : Quiver.Hom A✝ B✝
                    hf : CategoryTheory.IsIso ((CategoryTheory.forget SimplexCategory).map f)
                    y₁ y₂ : Fin (HAdd.hAdd B✝.len 1)
                    h : LE.le y₁ y₂
                    h' : LT.lt y₁ y₂
                    h'' : Not (LE.le (CategoryTheory.inv ((CategoryTheory.forget SimplexCategory). …
                    ⊢ LE.le y₂ y₁
                  -/
                  convert f.toOrderHom.monotone (le_of_not_ge h'')
                  all_goals
                    exact (congr_hom (Iso.inv_hom_id
                      (asIso ((forget SimplexCategory).map f))) _).symm
                  /-
                    case neg
                    A✝ B✝ : SimplexCategory
                    f : Quiver.Hom A✝ B✝
                    hf : CategoryTheory.IsIso ((CategoryTheory.forget SimplexCategory).map f)
                    y₁ y₂ : Fin (HAdd.hAdd B✝.len 1)
                    h : LE.le y₁ y₂
                    h' : Not (LT.lt y₁ y₂)
                    ⊢ LE.le (CategoryTheory.inv ((CategoryTheory.forget SimplexCategory).map f) y₁ …
                  -/
                · rw [eq_of_le_of_not_lt h h'] }
                  /-
                    🎉 no goals
                  -/
        hom_inv_id := by
          /-
            A✝ B✝ : SimplexCategory
            f : Quiver.Hom A✝ B✝
            hf : CategoryTheory.IsIso ((CategoryTheory.forget SimplexCategory).map f)
            ⊢ Eq (CategoryTheory.CategoryStruct.comp f (SimplexCategory.Hom.mk { toFun :=  …
          -/
          ext1
          /-
            case a
            A✝ B✝ : SimplexCategory
            f : Quiver.Hom A✝ B✝
            hf : CategoryTheory.IsIso ((CategoryTheory.forget SimplexCategory).map f)
            ⊢ Eq (SimplexCategory.Hom.toOrderHom (CategoryTheory.CategoryStruct.comp f (Si …
          -/
          ext1
          /-
            case a.h
            A✝ B✝ : SimplexCategory
            f : Quiver.Hom A✝ B✝
            hf : CategoryTheory.IsIso ((CategoryTheory.forget SimplexCategory).map f)
            ⊢ Eq ⇑(SimplexCategory.Hom.toOrderHom (CategoryTheory.CategoryStruct.comp f (S …
          -/
          exact Iso.hom_inv_id (asIso ((forget _).map f))
          /-
            🎉 no goals
          -/
        inv_hom_id := by
          /-
            A✝ B✝ : SimplexCategory
            f : Quiver.Hom A✝ B✝
            hf : CategoryTheory.IsIso ((CategoryTheory.forget SimplexCategory).map f)
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (SimplexCategory.Hom.mk { toFun := Ca …
          -/
          ext1
          /-
            case a
            A✝ B✝ : SimplexCategory
            f : Quiver.Hom A✝ B✝
            hf : CategoryTheory.IsIso ((CategoryTheory.forget SimplexCategory).map f)
            ⊢ Eq (SimplexCategory.Hom.toOrderHom (CategoryTheory.CategoryStruct.comp (Simp …
          -/
          ext1
          /-
            case a.h
            A✝ B✝ : SimplexCategory
            f : Quiver.Hom A✝ B✝
            hf : CategoryTheory.IsIso ((CategoryTheory.forget SimplexCategory).map f)
            ⊢ Eq ⇑(SimplexCategory.Hom.toOrderHom (CategoryTheory.CategoryStruct.comp (Sim …
          -/
          exact Iso.inv_hom_id (asIso ((forget _).map f)) }⟩
          /-
            🎉 no goals
          -/


theorem isIso_of_bijective {x y : SimplexCategory} {f : x ⟶ y}
    (hf : Function.Bijective f.toOrderHom.toFun) : IsIso f :=
  haveI : IsIso ((forget SimplexCategory).map f) := (isIso_iff_bijective _).mpr hf
  isIso_of_reflects_iso f (forget SimplexCategory)


/-- An isomorphism in `SimplexCategory` induces an `OrderIso`. -/
@[simp]
def orderIsoOfIso {x y : SimplexCategory} (e : x ≅ y) : Fin (x.len + 1) ≃o Fin (y.len + 1) :=
  Equiv.toOrderIso
    { toFun := e.hom.toOrderHom
      invFun := e.inv.toOrderHom
      left_inv := fun i => by
        /-
          x y : SimplexCategory
          e : CategoryTheory.Iso x y
          i : Fin (HAdd.hAdd x.len 1)
          ⊢ Eq ((SimplexCategory.Hom.toOrderHom e.inv) ((SimplexCategory.Hom.toOrderHom  …
        -/
        simpa only using congr_arg (fun φ => (Hom.toOrderHom φ) i) e.hom_inv_id
        /-
          🎉 no goals
        -/
      right_inv := fun i => by
        /-
          x y : SimplexCategory
          e : CategoryTheory.Iso x y
          i : Fin (HAdd.hAdd y.len 1)
          ⊢ Eq ((SimplexCategory.Hom.toOrderHom e.hom) ((SimplexCategory.Hom.toOrderHom  …
        -/
        simpa only using congr_arg (fun φ => (Hom.toOrderHom φ) i) e.inv_hom_id }
        /-
          🎉 no goals
        -/
    e.hom.toOrderHom.monotone e.inv.toOrderHom.monotone


theorem iso_eq_iso_refl {x : SimplexCategory} (e : x ≅ x) : e = Iso.refl x := by
  /-
    x : SimplexCategory
    e : CategoryTheory.Iso x x
    ⊢ Eq e (CategoryTheory.Iso.refl x)
  -/
  have h : (Finset.univ : Finset (Fin (x.len + 1))).card = x.len + 1 := Finset.card_fin (x.len + 1)
  /-
    x : SimplexCategory
    e : CategoryTheory.Iso x x
    h : Eq Finset.univ.card (HAdd.hAdd x.len 1)
    ⊢ Eq e (CategoryTheory.Iso.refl x)
  -/
  have eq₁ := Finset.orderEmbOfFin_unique' h fun i => Finset.mem_univ ((orderIsoOfIso e) i)
  have eq₂ :=
    Finset.orderEmbOfFin_unique' h fun i => Finset.mem_univ ((orderIsoOfIso (Iso.refl x)) i)
  -- Porting note: the proof was rewritten from this point in https://github.com/leanprover-community/mathlib4/pull/3414 (reenableeta)
  -- It could be investigated again to see if the original can be restored.
  /-
    x : SimplexCategory
    e : CategoryTheory.Iso x x
    h : Eq Finset.univ.card (HAdd.hAdd x.len 1)
    eq₁ : Eq (RelIso.toRelEmbedding (SimplexCategory.orderIsoOfIso e)) (Finset.uni …
    eq₂ : Eq (RelIso.toRelEmbedding (SimplexCategory.orderIsoOfIso (CategoryTheory …
    ⊢ Eq e (CategoryTheory.Iso.refl x)
  -/
  ext x
  /-
    case w.a.h.h.h
    x✝ : SimplexCategory
    e : CategoryTheory.Iso x✝ x✝
    h : Eq Finset.univ.card (HAdd.hAdd x✝.len 1)
    eq₁ : Eq (RelIso.toRelEmbedding (SimplexCategory.orderIsoOfIso e)) (Finset.uni …
    eq₂ : Eq (RelIso.toRelEmbedding (SimplexCategory.orderIsoOfIso (CategoryTheory …
    x : Fin (HAdd.hAdd x✝.len 1)
    ⊢ Eq ↑((SimplexCategory.Hom.toOrderHom e.hom) x) ↑((SimplexCategory.Hom.toOrde …
  -/
  replace eq₁ := congr_arg (· x) eq₁
  /-
    case w.a.h.h.h
    x✝ : SimplexCategory
    e : CategoryTheory.Iso x✝ x✝
    h : Eq Finset.univ.card (HAdd.hAdd x✝.len 1)
    eq₂ : Eq (RelIso.toRelEmbedding (SimplexCategory.orderIsoOfIso (CategoryTheory …
    x : Fin (HAdd.hAdd x✝.len 1)
    eq₁ : Eq ((fun x_1 => x_1 x) (RelIso.toRelEmbedding (SimplexCategory.orderIsoO …
    ⊢ Eq ↑((SimplexCategory.Hom.toOrderHom e.hom) x) ↑((SimplexCategory.Hom.toOrde …
  -/
  replace eq₂ := congr_arg (· x) eq₂.symm
  /-
    case w.a.h.h.h
    x✝ : SimplexCategory
    e : CategoryTheory.Iso x✝ x✝
    h : Eq Finset.univ.card (HAdd.hAdd x✝.len 1)
    x : Fin (HAdd.hAdd x✝.len 1)
    eq₁ : Eq ((fun x_1 => x_1 x) (RelIso.toRelEmbedding (SimplexCategory.orderIsoO …
    eq₂ : Eq ((fun x_1 => x_1 x) (Finset.univ.orderEmbOfFin h)) ((fun x_1 => x_1 x …
    ⊢ Eq ↑((SimplexCategory.Hom.toOrderHom e.hom) x) ↑((SimplexCategory.Hom.toOrde …
  -/
  simp_all
  /-
    🎉 no goals
  -/


theorem eq_id_of_isIso {x : SimplexCategory} (f : x ⟶ x) [IsIso f] : f = 𝟙 _ :=
  congr_arg (fun φ : _ ≅ _ => φ.hom) (iso_eq_iso_refl (asIso f))


theorem eq_σ_comp_of_not_injective' {n : ℕ} {Δ' : SimplexCategory} (θ : mk (n + 1) ⟶ Δ')
    (i : Fin (n + 1)) (hi : θ.toOrderHom (Fin.castSucc i) = θ.toOrderHom i.succ) :
    ∃ θ' : mk n ⟶ Δ', θ = σ i ≫ θ' := by
  /-
    n : Nat
    Δ' : SimplexCategory
    θ : Quiver.Hom (SimplexCategory.mk (HAdd.hAdd n 1)) Δ'
    i : Fin (HAdd.hAdd n 1)
    hi : Eq ((SimplexCategory.Hom.toOrderHom θ) i.castSucc) ((SimplexCategory.Hom. …
    ⊢ Exists fun θ' => Eq θ (CategoryTheory.CategoryStruct.comp (SimplexCategory.σ …
  -/
  use δ i.succ ≫ θ
  /-
    case h
    n : Nat
    Δ' : SimplexCategory
    θ : Quiver.Hom (SimplexCategory.mk (HAdd.hAdd n 1)) Δ'
    i : Fin (HAdd.hAdd n 1)
    hi : Eq ((SimplexCategory.Hom.toOrderHom θ) i.castSucc) ((SimplexCategory.Hom. …
    ⊢ Eq θ (CategoryTheory.CategoryStruct.comp (SimplexCategory.σ i) (CategoryTheo …
  -/
  ext1; ext1; ext1 x
  simp only [len_mk, σ, mkHom, comp_toOrderHom, Hom.toOrderHom_mk, OrderHom.comp_coe,
    OrderHom.coe_mk, Function.comp_apply]
  /-
    case h.a.h.h
    n : Nat
    Δ' : SimplexCategory
    θ : Quiver.Hom (SimplexCategory.mk (HAdd.hAdd n 1)) Δ'
    i : Fin (HAdd.hAdd n 1)
    hi : Eq ((SimplexCategory.Hom.toOrderHom θ) i.castSucc) ((SimplexCategory.Hom. …
    x : Fin (HAdd.hAdd (SimplexCategory.mk (HAdd.hAdd n 1)).len 1)
    ⊢ Eq ((SimplexCategory.Hom.toOrderHom θ) x) ((SimplexCategory.Hom.toOrderHom θ …
  -/
  by_cases h' : x ≤ Fin.castSucc i
  · -- This was not needed before https://github.com/leanprover/lean4/pull/2644
    /-
      case pos
      n : Nat
      Δ' : SimplexCategory
      θ : Quiver.Hom (SimplexCategory.mk (HAdd.hAdd n 1)) Δ'
      i : Fin (HAdd.hAdd n 1)
      hi : Eq ((SimplexCategory.Hom.toOrderHom θ) i.castSucc) ((SimplexCategory.Hom. …
      x : Fin (HAdd.hAdd (SimplexCategory.mk (HAdd.hAdd n 1)).len 1)
      h' : LE.le x i.castSucc
      ⊢ Eq ((SimplexCategory.Hom.toOrderHom θ) x) ((SimplexCategory.Hom.toOrderHom θ …
    -/
    dsimp
    /-
      case pos
      n : Nat
      Δ' : SimplexCategory
      θ : Quiver.Hom (SimplexCategory.mk (HAdd.hAdd n 1)) Δ'
      i : Fin (HAdd.hAdd n 1)
      hi : Eq ((SimplexCategory.Hom.toOrderHom θ) i.castSucc) ((SimplexCategory.Hom. …
      x : Fin (HAdd.hAdd (SimplexCategory.mk (HAdd.hAdd n 1)).len 1)
      h' : LE.le x i.castSucc
      ⊢ Eq ((SimplexCategory.Hom.toOrderHom θ) x) ((SimplexCategory.Hom.toOrderHom θ …
    -/
    rw [Fin.predAbove_of_le_castSucc i x h']
    /-
      case pos
      n : Nat
      Δ' : SimplexCategory
      θ : Quiver.Hom (SimplexCategory.mk (HAdd.hAdd n 1)) Δ'
      i : Fin (HAdd.hAdd n 1)
      hi : Eq ((SimplexCategory.Hom.toOrderHom θ) i.castSucc) ((SimplexCategory.Hom. …
      x : Fin (HAdd.hAdd (SimplexCategory.mk (HAdd.hAdd n 1)).len 1)
      h' : LE.le x i.castSucc
      ⊢ Eq ((SimplexCategory.Hom.toOrderHom θ) x) ((SimplexCategory.Hom.toOrderHom θ …
    -/
    dsimp [δ]
    /-
      case pos
      n : Nat
      Δ' : SimplexCategory
      θ : Quiver.Hom (SimplexCategory.mk (HAdd.hAdd n 1)) Δ'
      i : Fin (HAdd.hAdd n 1)
      hi : Eq ((SimplexCategory.Hom.toOrderHom θ) i.castSucc) ((SimplexCategory.Hom. …
      x : Fin (HAdd.hAdd (SimplexCategory.mk (HAdd.hAdd n 1)).len 1)
      h' : LE.le x i.castSucc
      ⊢ Eq ((SimplexCategory.Hom.toOrderHom θ) x) ((SimplexCategory.Hom.toOrderHom θ …
    -/
    rw [Fin.succAbove_of_castSucc_lt _ _ _]
      /-
        case pos
        n : Nat
        Δ' : SimplexCategory
        θ : Quiver.Hom (SimplexCategory.mk (HAdd.hAdd n 1)) Δ'
        i : Fin (HAdd.hAdd n 1)
        hi : Eq ((SimplexCategory.Hom.toOrderHom θ) i.castSucc) ((SimplexCategory.Hom. …
        x : Fin (HAdd.hAdd (SimplexCategory.mk (HAdd.hAdd n 1)).len 1)
        h' : LE.le x i.castSucc
        ⊢ Eq ((SimplexCategory.Hom.toOrderHom θ) x) ((SimplexCategory.Hom.toOrderHom θ …
      -/
    · rw [Fin.castSucc_castPred]
      /-
        🎉 no goals
      -/
      /-
        n : Nat
        Δ' : SimplexCategory
        θ : Quiver.Hom (SimplexCategory.mk (HAdd.hAdd n 1)) Δ'
        i : Fin (HAdd.hAdd n 1)
        hi : Eq ((SimplexCategory.Hom.toOrderHom θ) i.castSucc) ((SimplexCategory.Hom. …
        x : Fin (HAdd.hAdd (SimplexCategory.mk (HAdd.hAdd n 1)).len 1)
        h' : LE.le x i.castSucc
        ⊢ LT.lt (x.castPred ⋯).castSucc i.succ
      -/
    · exact (Fin.castSucc_lt_succ_iff.mpr h')
      /-
        🎉 no goals
      -/
    /-
      case neg
      n : Nat
      Δ' : SimplexCategory
      θ : Quiver.Hom (SimplexCategory.mk (HAdd.hAdd n 1)) Δ'
      i : Fin (HAdd.hAdd n 1)
      hi : Eq ((SimplexCategory.Hom.toOrderHom θ) i.castSucc) ((SimplexCategory.Hom. …
      x : Fin (HAdd.hAdd (SimplexCategory.mk (HAdd.hAdd n 1)).len 1)
      h' : Not (LE.le x i.castSucc)
      ⊢ Eq ((SimplexCategory.Hom.toOrderHom θ) x) ((SimplexCategory.Hom.toOrderHom θ …
    -/
  · simp only [not_le] at h'
    /-
      case neg
      n : Nat
      Δ' : SimplexCategory
      θ : Quiver.Hom (SimplexCategory.mk (HAdd.hAdd n 1)) Δ'
      i : Fin (HAdd.hAdd n 1)
      hi : Eq ((SimplexCategory.Hom.toOrderHom θ) i.castSucc) ((SimplexCategory.Hom. …
      x : Fin (HAdd.hAdd (SimplexCategory.mk (HAdd.hAdd n 1)).len 1)
      h' : LT.lt i.castSucc x
      ⊢ Eq ((SimplexCategory.Hom.toOrderHom θ) x) ((SimplexCategory.Hom.toOrderHom θ …
    -/
    let y := x.pred <| by rintro (rfl : x = 0); simp at h'
    /-
      case neg
      n : Nat
      Δ' : SimplexCategory
      θ : Quiver.Hom (SimplexCategory.mk (HAdd.hAdd n 1)) Δ'
      i : Fin (HAdd.hAdd n 1)
      hi : Eq ((SimplexCategory.Hom.toOrderHom θ) i.castSucc) ((SimplexCategory.Hom. …
      x : Fin (HAdd.hAdd (SimplexCategory.mk (HAdd.hAdd n 1)).len 1)
      h' : LT.lt i.castSucc x
      y : Fin (SimplexCategory.mk (HAdd.hAdd n 1)).len := x.pred ⋯
      ⊢ Eq ((SimplexCategory.Hom.toOrderHom θ) x) ((SimplexCategory.Hom.toOrderHom θ …
    -/
    have hy : x = y.succ := (Fin.succ_pred x _).symm
    /-
      case neg
      n : Nat
      Δ' : SimplexCategory
      θ : Quiver.Hom (SimplexCategory.mk (HAdd.hAdd n 1)) Δ'
      i : Fin (HAdd.hAdd n 1)
      hi : Eq ((SimplexCategory.Hom.toOrderHom θ) i.castSucc) ((SimplexCategory.Hom. …
      x : Fin (HAdd.hAdd (SimplexCategory.mk (HAdd.hAdd n 1)).len 1)
      h' : LT.lt i.castSucc x
      y : Fin (SimplexCategory.mk (HAdd.hAdd n 1)).len := x.pred ⋯
      hy : Eq x y.succ
      ⊢ Eq ((SimplexCategory.Hom.toOrderHom θ) x) ((SimplexCategory.Hom.toOrderHom θ …
    -/
    rw [hy] at h' ⊢
    -- This was not needed before https://github.com/leanprover/lean4/pull/2644
    /-
      case neg
      n : Nat
      Δ' : SimplexCategory
      θ : Quiver.Hom (SimplexCategory.mk (HAdd.hAdd n 1)) Δ'
      i : Fin (HAdd.hAdd n 1)
      hi : Eq ((SimplexCategory.Hom.toOrderHom θ) i.castSucc) ((SimplexCategory.Hom. …
      x : Fin (HAdd.hAdd (SimplexCategory.mk (HAdd.hAdd n 1)).len 1)
      h'✝ : LT.lt i.castSucc x
      y : Fin (SimplexCategory.mk (HAdd.hAdd n 1)).len := x.pred ⋯
      h' : LT.lt i.castSucc y.succ
      hy : Eq x y.succ
      ⊢ Eq ((SimplexCategory.Hom.toOrderHom θ) y.succ) ((SimplexCategory.Hom.toOrder …
    -/
    conv_rhs => dsimp
    /-
      case neg
      n : Nat
      Δ' : SimplexCategory
      θ : Quiver.Hom (SimplexCategory.mk (HAdd.hAdd n 1)) Δ'
      i : Fin (HAdd.hAdd n 1)
      hi : Eq ((SimplexCategory.Hom.toOrderHom θ) i.castSucc) ((SimplexCategory.Hom. …
      x : Fin (HAdd.hAdd (SimplexCategory.mk (HAdd.hAdd n 1)).len 1)
      h'✝ : LT.lt i.castSucc x
      y : Fin (SimplexCategory.mk (HAdd.hAdd n 1)).len := x.pred ⋯
      h' : LT.lt i.castSucc y.succ
      hy : Eq x y.succ
      ⊢ Eq ((SimplexCategory.Hom.toOrderHom θ) y.succ) ((SimplexCategory.Hom.toOrder …
    -/
    rw [Fin.predAbove_of_castSucc_lt i y.succ h', Fin.pred_succ]
    /-
      case neg
      n : Nat
      Δ' : SimplexCategory
      θ : Quiver.Hom (SimplexCategory.mk (HAdd.hAdd n 1)) Δ'
      i : Fin (HAdd.hAdd n 1)
      hi : Eq ((SimplexCategory.Hom.toOrderHom θ) i.castSucc) ((SimplexCategory.Hom. …
      x : Fin (HAdd.hAdd (SimplexCategory.mk (HAdd.hAdd n 1)).len 1)
      h'✝ : LT.lt i.castSucc x
      y : Fin (SimplexCategory.mk (HAdd.hAdd n 1)).len := x.pred ⋯
      h' : LT.lt i.castSucc y.succ
      hy : Eq x y.succ
      ⊢ Eq ((SimplexCategory.Hom.toOrderHom θ) y.succ) ((SimplexCategory.Hom.toOrder …
    -/
    by_cases h'' : y = i
      /-
        case pos
        n : Nat
        Δ' : SimplexCategory
        θ : Quiver.Hom (SimplexCategory.mk (HAdd.hAdd n 1)) Δ'
        i : Fin (HAdd.hAdd n 1)
        hi : Eq ((SimplexCategory.Hom.toOrderHom θ) i.castSucc) ((SimplexCategory.Hom. …
        x : Fin (HAdd.hAdd (SimplexCategory.mk (HAdd.hAdd n 1)).len 1)
        h'✝ : LT.lt i.castSucc x
        y : Fin (SimplexCategory.mk (HAdd.hAdd n 1)).len := x.pred ⋯
        h' : LT.lt i.castSucc y.succ
        hy : Eq x y.succ
        h'' : Eq y i
        ⊢ Eq ((SimplexCategory.Hom.toOrderHom θ) y.succ) ((SimplexCategory.Hom.toOrder …
      -/
    · rw [h'']
      /-
        case pos
        n : Nat
        Δ' : SimplexCategory
        θ : Quiver.Hom (SimplexCategory.mk (HAdd.hAdd n 1)) Δ'
        i : Fin (HAdd.hAdd n 1)
        hi : Eq ((SimplexCategory.Hom.toOrderHom θ) i.castSucc) ((SimplexCategory.Hom. …
        x : Fin (HAdd.hAdd (SimplexCategory.mk (HAdd.hAdd n 1)).len 1)
        h'✝ : LT.lt i.castSucc x
        y : Fin (SimplexCategory.mk (HAdd.hAdd n 1)).len := x.pred ⋯
        h' : LT.lt i.castSucc y.succ
        hy : Eq x y.succ
        h'' : Eq y i
        ⊢ Eq ((SimplexCategory.Hom.toOrderHom θ) i.succ) ((SimplexCategory.Hom.toOrder …
      -/
      refine hi.symm.trans ?_
      /-
        case pos
        n : Nat
        Δ' : SimplexCategory
        θ : Quiver.Hom (SimplexCategory.mk (HAdd.hAdd n 1)) Δ'
        i : Fin (HAdd.hAdd n 1)
        hi : Eq ((SimplexCategory.Hom.toOrderHom θ) i.castSucc) ((SimplexCategory.Hom. …
        x : Fin (HAdd.hAdd (SimplexCategory.mk (HAdd.hAdd n 1)).len 1)
        h'✝ : LT.lt i.castSucc x
        y : Fin (SimplexCategory.mk (HAdd.hAdd n 1)).len := x.pred ⋯
        h' : LT.lt i.castSucc y.succ
        hy : Eq x y.succ
        h'' : Eq y i
        ⊢ Eq ((SimplexCategory.Hom.toOrderHom θ) i.castSucc) ((SimplexCategory.Hom.toO …
      -/
      congr 1
      /-
        case pos.h.e_6.h
        n : Nat
        Δ' : SimplexCategory
        θ : Quiver.Hom (SimplexCategory.mk (HAdd.hAdd n 1)) Δ'
        i : Fin (HAdd.hAdd n 1)
        hi : Eq ((SimplexCategory.Hom.toOrderHom θ) i.castSucc) ((SimplexCategory.Hom. …
        x : Fin (HAdd.hAdd (SimplexCategory.mk (HAdd.hAdd n 1)).len 1)
        h'✝ : LT.lt i.castSucc x
        y : Fin (SimplexCategory.mk (HAdd.hAdd n 1)).len := x.pred ⋯
        h' : LT.lt i.castSucc y.succ
        hy : Eq x y.succ
        h'' : Eq y i
        ⊢ Eq i.castSucc ((SimplexCategory.Hom.toOrderHom (SimplexCategory.δ i.succ)) i)
      -/
      dsimp [δ]
      /-
        case pos.h.e_6.h
        n : Nat
        Δ' : SimplexCategory
        θ : Quiver.Hom (SimplexCategory.mk (HAdd.hAdd n 1)) Δ'
        i : Fin (HAdd.hAdd n 1)
        hi : Eq ((SimplexCategory.Hom.toOrderHom θ) i.castSucc) ((SimplexCategory.Hom. …
        x : Fin (HAdd.hAdd (SimplexCategory.mk (HAdd.hAdd n 1)).len 1)
        h'✝ : LT.lt i.castSucc x
        y : Fin (SimplexCategory.mk (HAdd.hAdd n 1)).len := x.pred ⋯
        h' : LT.lt i.castSucc y.succ
        hy : Eq x y.succ
        h'' : Eq y i
        ⊢ Eq i.castSucc (i.succ.succAbove i)
      -/
      rw [Fin.succAbove_of_castSucc_lt i.succ]
      /-
        case pos.h.e_6.h.h
        n : Nat
        Δ' : SimplexCategory
        θ : Quiver.Hom (SimplexCategory.mk (HAdd.hAdd n 1)) Δ'
        i : Fin (HAdd.hAdd n 1)
        hi : Eq ((SimplexCategory.Hom.toOrderHom θ) i.castSucc) ((SimplexCategory.Hom. …
        x : Fin (HAdd.hAdd (SimplexCategory.mk (HAdd.hAdd n 1)).len 1)
        h'✝ : LT.lt i.castSucc x
        y : Fin (SimplexCategory.mk (HAdd.hAdd n 1)).len := x.pred ⋯
        h' : LT.lt i.castSucc y.succ
        hy : Eq x y.succ
        h'' : Eq y i
        ⊢ LT.lt i.castSucc i.succ
      -/
      exact Fin.lt_succ
      /-
        🎉 no goals
      -/
      /-
        case neg
        n : Nat
        Δ' : SimplexCategory
        θ : Quiver.Hom (SimplexCategory.mk (HAdd.hAdd n 1)) Δ'
        i : Fin (HAdd.hAdd n 1)
        hi : Eq ((SimplexCategory.Hom.toOrderHom θ) i.castSucc) ((SimplexCategory.Hom. …
        x : Fin (HAdd.hAdd (SimplexCategory.mk (HAdd.hAdd n 1)).len 1)
        h'✝ : LT.lt i.castSucc x
        y : Fin (SimplexCategory.mk (HAdd.hAdd n 1)).len := x.pred ⋯
        h' : LT.lt i.castSucc y.succ
        hy : Eq x y.succ
        h'' : Not (Eq y i)
        ⊢ Eq ((SimplexCategory.Hom.toOrderHom θ) y.succ) ((SimplexCategory.Hom.toOrder …
      -/
    · dsimp [δ]
      /-
        case neg
        n : Nat
        Δ' : SimplexCategory
        θ : Quiver.Hom (SimplexCategory.mk (HAdd.hAdd n 1)) Δ'
        i : Fin (HAdd.hAdd n 1)
        hi : Eq ((SimplexCategory.Hom.toOrderHom θ) i.castSucc) ((SimplexCategory.Hom. …
        x : Fin (HAdd.hAdd (SimplexCategory.mk (HAdd.hAdd n 1)).len 1)
        h'✝ : LT.lt i.castSucc x
        y : Fin (SimplexCategory.mk (HAdd.hAdd n 1)).len := x.pred ⋯
        h' : LT.lt i.castSucc y.succ
        hy : Eq x y.succ
        h'' : Not (Eq y i)
        ⊢ Eq ((SimplexCategory.Hom.toOrderHom θ) y.succ) ((SimplexCategory.Hom.toOrder …
      -/
      rw [Fin.succAbove_of_le_castSucc i.succ _]
      simp only [Fin.lt_iff_val_lt_val, Fin.le_iff_val_le_val, Fin.val_succ, Fin.coe_castSucc,
        Nat.lt_succ_iff, Fin.ext_iff] at h' h'' ⊢
      /-
        case neg
        n : Nat
        Δ' : SimplexCategory
        θ : Quiver.Hom (SimplexCategory.mk (HAdd.hAdd n 1)) Δ'
        i : Fin (HAdd.hAdd n 1)
        hi : Eq ((SimplexCategory.Hom.toOrderHom θ) i.castSucc) ((SimplexCategory.Hom. …
        x : Fin (HAdd.hAdd (SimplexCategory.mk (HAdd.hAdd n 1)).len 1)
        h'✝ : LT.lt i.castSucc x
        y : Fin (SimplexCategory.mk (HAdd.hAdd n 1)).len := x.pred ⋯
        hy : Eq x y.succ
        h' : LE.le ↑i ↑y
        h'' : Not (Eq ↑y ↑i)
        ⊢ LE.le (HAdd.hAdd (↑i) 1) ↑y
      -/
      cases' Nat.le.dest h' with c hc
      /-
        case neg.intro
        n : Nat
        Δ' : SimplexCategory
        θ : Quiver.Hom (SimplexCategory.mk (HAdd.hAdd n 1)) Δ'
        i : Fin (HAdd.hAdd n 1)
        hi : Eq ((SimplexCategory.Hom.toOrderHom θ) i.castSucc) ((SimplexCategory.Hom. …
        x : Fin (HAdd.hAdd (SimplexCategory.mk (HAdd.hAdd n 1)).len 1)
        h'✝ : LT.lt i.castSucc x
        y : Fin (SimplexCategory.mk (HAdd.hAdd n 1)).len := x.pred ⋯
        hy : Eq x y.succ
        h' : LE.le ↑i ↑y
        h'' : Not (Eq ↑y ↑i)
        c : Nat
        hc : Eq (HAdd.hAdd (↑i) c) ↑y
        ⊢ LE.le (HAdd.hAdd (↑i) 1) ↑y
      -/
      cases c
        /-
          case neg.intro.zero
          n : Nat
          Δ' : SimplexCategory
          θ : Quiver.Hom (SimplexCategory.mk (HAdd.hAdd n 1)) Δ'
          i : Fin (HAdd.hAdd n 1)
          hi : Eq ((SimplexCategory.Hom.toOrderHom θ) i.castSucc) ((SimplexCategory.Hom. …
          x : Fin (HAdd.hAdd (SimplexCategory.mk (HAdd.hAdd n 1)).len 1)
          h'✝ : LT.lt i.castSucc x
          y : Fin (SimplexCategory.mk (HAdd.hAdd n 1)).len := x.pred ⋯
          hy : Eq x y.succ
          h' : LE.le ↑i ↑y
          h'' : Not (Eq ↑y ↑i)
          hc : Eq (HAdd.hAdd (↑i) 0) ↑y
          ⊢ LE.le (HAdd.hAdd (↑i) 1) ↑y
        -/
      · exfalso
        /-
          case neg.intro.zero
          n : Nat
          Δ' : SimplexCategory
          θ : Quiver.Hom (SimplexCategory.mk (HAdd.hAdd n 1)) Δ'
          i : Fin (HAdd.hAdd n 1)
          hi : Eq ((SimplexCategory.Hom.toOrderHom θ) i.castSucc) ((SimplexCategory.Hom. …
          x : Fin (HAdd.hAdd (SimplexCategory.mk (HAdd.hAdd n 1)).len 1)
          h'✝ : LT.lt i.castSucc x
          y : Fin (SimplexCategory.mk (HAdd.hAdd n 1)).len := x.pred ⋯
          hy : Eq x y.succ
          h' : LE.le ↑i ↑y
          h'' : Not (Eq ↑y ↑i)
          hc : Eq (HAdd.hAdd (↑i) 0) ↑y
          ⊢ False
        -/
        simp only [add_zero, len_mk, Fin.coe_pred] at hc
        /-
          case neg.intro.zero
          n : Nat
          Δ' : SimplexCategory
          θ : Quiver.Hom (SimplexCategory.mk (HAdd.hAdd n 1)) Δ'
          i : Fin (HAdd.hAdd n 1)
          hi : Eq ((SimplexCategory.Hom.toOrderHom θ) i.castSucc) ((SimplexCategory.Hom. …
          x : Fin (HAdd.hAdd (SimplexCategory.mk (HAdd.hAdd n 1)).len 1)
          h'✝ : LT.lt i.castSucc x
          y : Fin (SimplexCategory.mk (HAdd.hAdd n 1)).len := x.pred ⋯
          hy : Eq x y.succ
          h' : LE.le ↑i ↑y
          h'' : Not (Eq ↑y ↑i)
          hc : Eq ↑i ↑y
          ⊢ False
        -/
        rw [hc] at h''
        /-
          case neg.intro.zero
          n : Nat
          Δ' : SimplexCategory
          θ : Quiver.Hom (SimplexCategory.mk (HAdd.hAdd n 1)) Δ'
          i : Fin (HAdd.hAdd n 1)
          hi : Eq ((SimplexCategory.Hom.toOrderHom θ) i.castSucc) ((SimplexCategory.Hom. …
          x : Fin (HAdd.hAdd (SimplexCategory.mk (HAdd.hAdd n 1)).len 1)
          h'✝ : LT.lt i.castSucc x
          y : Fin (SimplexCategory.mk (HAdd.hAdd n 1)).len := x.pred ⋯
          hy : Eq x y.succ
          h' : LE.le ↑i ↑y
          h'' : Not (Eq ↑y ↑y)
          hc : Eq ↑i ↑y
          ⊢ False
        -/
        exact h'' rfl
        /-
          🎉 no goals
        -/
        /-
          case neg.intro.succ
          n : Nat
          Δ' : SimplexCategory
          θ : Quiver.Hom (SimplexCategory.mk (HAdd.hAdd n 1)) Δ'
          i : Fin (HAdd.hAdd n 1)
          hi : Eq ((SimplexCategory.Hom.toOrderHom θ) i.castSucc) ((SimplexCategory.Hom. …
          x : Fin (HAdd.hAdd (SimplexCategory.mk (HAdd.hAdd n 1)).len 1)
          h'✝ : LT.lt i.castSucc x
          y : Fin (SimplexCategory.mk (HAdd.hAdd n 1)).len := x.pred ⋯
          hy : Eq x y.succ
          h' : LE.le ↑i ↑y
          h'' : Not (Eq ↑y ↑i)
          n✝ : Nat
          hc : Eq (HAdd.hAdd (↑i) (HAdd.hAdd n✝ 1)) ↑y
          ⊢ LE.le (HAdd.hAdd (↑i) 1) ↑y
        -/
      · rw [← hc]
        /-
          case neg.intro.succ
          n : Nat
          Δ' : SimplexCategory
          θ : Quiver.Hom (SimplexCategory.mk (HAdd.hAdd n 1)) Δ'
          i : Fin (HAdd.hAdd n 1)
          hi : Eq ((SimplexCategory.Hom.toOrderHom θ) i.castSucc) ((SimplexCategory.Hom. …
          x : Fin (HAdd.hAdd (SimplexCategory.mk (HAdd.hAdd n 1)).len 1)
          h'✝ : LT.lt i.castSucc x
          y : Fin (SimplexCategory.mk (HAdd.hAdd n 1)).len := x.pred ⋯
          hy : Eq x y.succ
          h' : LE.le ↑i ↑y
          h'' : Not (Eq ↑y ↑i)
          n✝ : Nat
          hc : Eq (HAdd.hAdd (↑i) (HAdd.hAdd n✝ 1)) ↑y
          ⊢ LE.le (HAdd.hAdd (↑i) 1) (HAdd.hAdd (↑i) (HAdd.hAdd n✝ 1))
        -/
        simp only [add_le_add_iff_left, Nat.succ_eq_add_one, le_add_iff_nonneg_left, zero_le]
        /-
          🎉 no goals
        -/


theorem eq_σ_comp_of_not_injective {n : ℕ} {Δ' : SimplexCategory} (θ : mk (n + 1) ⟶ Δ')
    (hθ : ¬Function.Injective θ.toOrderHom) :
    ∃ (i : Fin (n + 1)) (θ' : mk n ⟶ Δ'), θ = σ i ≫ θ' := by
  /-
    n : Nat
    Δ' : SimplexCategory
    θ : Quiver.Hom (SimplexCategory.mk (HAdd.hAdd n 1)) Δ'
    hθ : Not (Function.Injective ⇑(SimplexCategory.Hom.toOrderHom θ))
    ⊢ Exists fun i => Exists fun θ' => Eq θ (CategoryTheory.CategoryStruct.comp (S …
  -/
  simp only [Function.Injective, exists_prop, not_forall] at hθ
  -- as θ is not injective, there exists `x<y` such that `θ x = θ y`
  -- and then, `θ x = θ (x+1)`
  have hθ₂ : ∃ x y : Fin (n + 2), (Hom.toOrderHom θ) x = (Hom.toOrderHom θ) y ∧ x < y := by
    rcases hθ with ⟨x, y, ⟨h₁, h₂⟩⟩
    by_cases h : x < y
    · exact ⟨x, y, ⟨h₁, h⟩⟩
    · refine ⟨y, x, ⟨h₁.symm, ?_⟩⟩
      rcases lt_or_eq_of_le (not_lt.mp h) with h' | h'
      · exact h'
      · exfalso
        exact h₂ h'.symm
  /-
    n : Nat
    Δ' : SimplexCategory
    θ : Quiver.Hom (SimplexCategory.mk (HAdd.hAdd n 1)) Δ'
    hθ : Exists fun x => Exists fun x_1 => And (Eq ((SimplexCategory.Hom.toOrderHo …
    hθ₂ : Exists fun x => Exists fun y => And (Eq ((SimplexCategory.Hom.toOrderHom …
    ⊢ Exists fun i => Exists fun θ' => Eq θ (CategoryTheory.CategoryStruct.comp (S …
  -/
  rcases hθ₂ with ⟨x, y, ⟨h₁, h₂⟩⟩
  /-
    case intro.intro.intro
    n : Nat
    Δ' : SimplexCategory
    θ : Quiver.Hom (SimplexCategory.mk (HAdd.hAdd n 1)) Δ'
    hθ : Exists fun x => Exists fun x_1 => And (Eq ((SimplexCategory.Hom.toOrderHo …
    x y : Fin (HAdd.hAdd n 2)
    h₁ : Eq ((SimplexCategory.Hom.toOrderHom θ) x) ((SimplexCategory.Hom.toOrderHo …
    h₂ : LT.lt x y
    ⊢ Exists fun i => Exists fun θ' => Eq θ (CategoryTheory.CategoryStruct.comp (S …
  -/
  use x.castPred ((Fin.le_last _).trans_lt' h₂).ne
  /-
    case h
    n : Nat
    Δ' : SimplexCategory
    θ : Quiver.Hom (SimplexCategory.mk (HAdd.hAdd n 1)) Δ'
    hθ : Exists fun x => Exists fun x_1 => And (Eq ((SimplexCategory.Hom.toOrderHo …
    x y : Fin (HAdd.hAdd n 2)
    h₁ : Eq ((SimplexCategory.Hom.toOrderHom θ) x) ((SimplexCategory.Hom.toOrderHo …
    h₂ : LT.lt x y
    ⊢ Exists fun θ' => Eq θ (CategoryTheory.CategoryStruct.comp (SimplexCategory.σ …
  -/
  apply eq_σ_comp_of_not_injective'
  /-
    case h.hi
    n : Nat
    Δ' : SimplexCategory
    θ : Quiver.Hom (SimplexCategory.mk (HAdd.hAdd n 1)) Δ'
    hθ : Exists fun x => Exists fun x_1 => And (Eq ((SimplexCategory.Hom.toOrderHo …
    x y : Fin (HAdd.hAdd n 2)
    h₁ : Eq ((SimplexCategory.Hom.toOrderHom θ) x) ((SimplexCategory.Hom.toOrderHo …
    h₂ : LT.lt x y
    ⊢ Eq ((SimplexCategory.Hom.toOrderHom θ) (x.castPred ⋯).castSucc) ((SimplexCat …
  -/
  apply le_antisymm
    /-
      case h.hi.a
      n : Nat
      Δ' : SimplexCategory
      θ : Quiver.Hom (SimplexCategory.mk (HAdd.hAdd n 1)) Δ'
      hθ : Exists fun x => Exists fun x_1 => And (Eq ((SimplexCategory.Hom.toOrderHo …
      x y : Fin (HAdd.hAdd n 2)
      h₁ : Eq ((SimplexCategory.Hom.toOrderHom θ) x) ((SimplexCategory.Hom.toOrderHo …
      h₂ : LT.lt x y
      ⊢ LE.le ((SimplexCategory.Hom.toOrderHom θ) (x.castPred ⋯).castSucc) ((Simplex …
    -/
  · exact θ.toOrderHom.monotone (le_of_lt (Fin.castSucc_lt_succ _))
    /-
      🎉 no goals
    -/
    /-
      case h.hi.a
      n : Nat
      Δ' : SimplexCategory
      θ : Quiver.Hom (SimplexCategory.mk (HAdd.hAdd n 1)) Δ'
      hθ : Exists fun x => Exists fun x_1 => And (Eq ((SimplexCategory.Hom.toOrderHo …
      x y : Fin (HAdd.hAdd n 2)
      h₁ : Eq ((SimplexCategory.Hom.toOrderHom θ) x) ((SimplexCategory.Hom.toOrderHo …
      h₂ : LT.lt x y
      ⊢ LE.le ((SimplexCategory.Hom.toOrderHom θ) (x.castPred ⋯).succ) ((SimplexCate …
    -/
  · rw [Fin.castSucc_castPred, h₁]
    /-
      case h.hi.a
      n : Nat
      Δ' : SimplexCategory
      θ : Quiver.Hom (SimplexCategory.mk (HAdd.hAdd n 1)) Δ'
      hθ : Exists fun x => Exists fun x_1 => And (Eq ((SimplexCategory.Hom.toOrderHo …
      x y : Fin (HAdd.hAdd n 2)
      h₁ : Eq ((SimplexCategory.Hom.toOrderHom θ) x) ((SimplexCategory.Hom.toOrderHo …
      h₂ : LT.lt x y
      ⊢ LE.le ((SimplexCategory.Hom.toOrderHom θ) (x.castPred ⋯).succ) ((SimplexCate …
    -/
    exact θ.toOrderHom.monotone ((Fin.succ_castPred_le_iff _).mpr h₂)
    /-
      🎉 no goals
    -/


theorem eq_comp_δ_of_not_surjective' {n : ℕ} {Δ : SimplexCategory} (θ : Δ ⟶ mk (n + 1))
    (i : Fin (n + 2)) (hi : ∀ x, θ.toOrderHom x ≠ i) : ∃ θ' : Δ ⟶ mk n, θ = θ' ≫ δ i := by
  /-
    n : Nat
    Δ : SimplexCategory
    θ : Quiver.Hom Δ (SimplexCategory.mk (HAdd.hAdd n 1))
    i : Fin (HAdd.hAdd n 2)
    hi : ∀ (x : Fin (HAdd.hAdd Δ.len 1)), Ne ((SimplexCategory.Hom.toOrderHom θ) x …
    ⊢ Exists fun θ' => Eq θ (CategoryTheory.CategoryStruct.comp θ' (SimplexCategor …
  -/
  by_cases h : i < Fin.last (n + 1)
    /-
      case pos
      n : Nat
      Δ : SimplexCategory
      θ : Quiver.Hom Δ (SimplexCategory.mk (HAdd.hAdd n 1))
      i : Fin (HAdd.hAdd n 2)
      hi : ∀ (x : Fin (HAdd.hAdd Δ.len 1)), Ne ((SimplexCategory.Hom.toOrderHom θ) x …
      h : LT.lt i (Fin.last (HAdd.hAdd n 1))
      ⊢ Exists fun θ' => Eq θ (CategoryTheory.CategoryStruct.comp θ' (SimplexCategor …
    -/
  · use θ ≫ σ (Fin.castPred i h.ne)
    /-
      case h
      n : Nat
      Δ : SimplexCategory
      θ : Quiver.Hom Δ (SimplexCategory.mk (HAdd.hAdd n 1))
      i : Fin (HAdd.hAdd n 2)
      hi : ∀ (x : Fin (HAdd.hAdd Δ.len 1)), Ne ((SimplexCategory.Hom.toOrderHom θ) x …
      h : LT.lt i (Fin.last (HAdd.hAdd n 1))
      ⊢ Eq θ (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp …
    -/
    ext1
    /-
      case h.a
      n : Nat
      Δ : SimplexCategory
      θ : Quiver.Hom Δ (SimplexCategory.mk (HAdd.hAdd n 1))
      i : Fin (HAdd.hAdd n 2)
      hi : ∀ (x : Fin (HAdd.hAdd Δ.len 1)), Ne ((SimplexCategory.Hom.toOrderHom θ) x …
      h : LT.lt i (Fin.last (HAdd.hAdd n 1))
      ⊢ Eq (SimplexCategory.Hom.toOrderHom θ) (SimplexCategory.Hom.toOrderHom (Categ …
    -/
    ext1
    /-
      case h.a.h
      n : Nat
      Δ : SimplexCategory
      θ : Quiver.Hom Δ (SimplexCategory.mk (HAdd.hAdd n 1))
      i : Fin (HAdd.hAdd n 2)
      hi : ∀ (x : Fin (HAdd.hAdd Δ.len 1)), Ne ((SimplexCategory.Hom.toOrderHom θ) x …
      h : LT.lt i (Fin.last (HAdd.hAdd n 1))
      ⊢ Eq ⇑(SimplexCategory.Hom.toOrderHom θ) ⇑(SimplexCategory.Hom.toOrderHom (Cat …
    -/
    ext1 x
    /-
      case h.a.h.h
      n : Nat
      Δ : SimplexCategory
      θ : Quiver.Hom Δ (SimplexCategory.mk (HAdd.hAdd n 1))
      i : Fin (HAdd.hAdd n 2)
      hi : ∀ (x : Fin (HAdd.hAdd Δ.len 1)), Ne ((SimplexCategory.Hom.toOrderHom θ) x …
      h : LT.lt i (Fin.last (HAdd.hAdd n 1))
      x : Fin (HAdd.hAdd Δ.len 1)
      ⊢ Eq ((SimplexCategory.Hom.toOrderHom θ) x) ((SimplexCategory.Hom.toOrderHom ( …
    -/
    simp only [len_mk, Category.assoc, comp_toOrderHom, OrderHom.comp_coe, Function.comp_apply]
    /-
      case h.a.h.h
      n : Nat
      Δ : SimplexCategory
      θ : Quiver.Hom Δ (SimplexCategory.mk (HAdd.hAdd n 1))
      i : Fin (HAdd.hAdd n 2)
      hi : ∀ (x : Fin (HAdd.hAdd Δ.len 1)), Ne ((SimplexCategory.Hom.toOrderHom θ) x …
      h : LT.lt i (Fin.last (HAdd.hAdd n 1))
      x : Fin (HAdd.hAdd Δ.len 1)
      ⊢ Eq ((SimplexCategory.Hom.toOrderHom θ) x) ((SimplexCategory.Hom.toOrderHom ( …
    -/
    by_cases h' : θ.toOrderHom x ≤ i
      /-
        case pos
        n : Nat
        Δ : SimplexCategory
        θ : Quiver.Hom Δ (SimplexCategory.mk (HAdd.hAdd n 1))
        i : Fin (HAdd.hAdd n 2)
        hi : ∀ (x : Fin (HAdd.hAdd Δ.len 1)), Ne ((SimplexCategory.Hom.toOrderHom θ) x …
        h : LT.lt i (Fin.last (HAdd.hAdd n 1))
        x : Fin (HAdd.hAdd Δ.len 1)
        h' : LE.le ((SimplexCategory.Hom.toOrderHom θ) x) i
        ⊢ Eq ((SimplexCategory.Hom.toOrderHom θ) x) ((SimplexCategory.Hom.toOrderHom ( …
      -/
    · simp only [σ, mkHom, Hom.toOrderHom_mk, OrderHom.coe_mk]
      /-
        case pos
        n : Nat
        Δ : SimplexCategory
        θ : Quiver.Hom Δ (SimplexCategory.mk (HAdd.hAdd n 1))
        i : Fin (HAdd.hAdd n 2)
        hi : ∀ (x : Fin (HAdd.hAdd Δ.len 1)), Ne ((SimplexCategory.Hom.toOrderHom θ) x …
        h : LT.lt i (Fin.last (HAdd.hAdd n 1))
        x : Fin (HAdd.hAdd Δ.len 1)
        h' : LE.le ((SimplexCategory.Hom.toOrderHom θ) x) i
        ⊢ Eq ((SimplexCategory.Hom.toOrderHom θ) x) ((SimplexCategory.Hom.toOrderHom ( …
      -/
      rw [Fin.predAbove_of_le_castSucc _ _ (by rwa [Fin.castSucc_castPred])]
      /-
        case pos
        n : Nat
        Δ : SimplexCategory
        θ : Quiver.Hom Δ (SimplexCategory.mk (HAdd.hAdd n 1))
        i : Fin (HAdd.hAdd n 2)
        hi : ∀ (x : Fin (HAdd.hAdd Δ.len 1)), Ne ((SimplexCategory.Hom.toOrderHom θ) x …
        h : LT.lt i (Fin.last (HAdd.hAdd n 1))
        x : Fin (HAdd.hAdd Δ.len 1)
        h' : LE.le ((SimplexCategory.Hom.toOrderHom θ) x) i
        ⊢ Eq ((SimplexCategory.Hom.toOrderHom θ) x) ((SimplexCategory.Hom.toOrderHom ( …
      -/
      dsimp [δ]
      /-
        case pos
        n : Nat
        Δ : SimplexCategory
        θ : Quiver.Hom Δ (SimplexCategory.mk (HAdd.hAdd n 1))
        i : Fin (HAdd.hAdd n 2)
        hi : ∀ (x : Fin (HAdd.hAdd Δ.len 1)), Ne ((SimplexCategory.Hom.toOrderHom θ) x …
        h : LT.lt i (Fin.last (HAdd.hAdd n 1))
        x : Fin (HAdd.hAdd Δ.len 1)
        h' : LE.le ((SimplexCategory.Hom.toOrderHom θ) x) i
        ⊢ Eq ((SimplexCategory.Hom.toOrderHom θ) x) (i.succAbove (((SimplexCategory.Ho …
      -/
      rw [Fin.succAbove_of_castSucc_lt i]
        /-
          case pos
          n : Nat
          Δ : SimplexCategory
          θ : Quiver.Hom Δ (SimplexCategory.mk (HAdd.hAdd n 1))
          i : Fin (HAdd.hAdd n 2)
          hi : ∀ (x : Fin (HAdd.hAdd Δ.len 1)), Ne ((SimplexCategory.Hom.toOrderHom θ) x …
          h : LT.lt i (Fin.last (HAdd.hAdd n 1))
          x : Fin (HAdd.hAdd Δ.len 1)
          h' : LE.le ((SimplexCategory.Hom.toOrderHom θ) x) i
          ⊢ Eq ((SimplexCategory.Hom.toOrderHom θ) x) (((SimplexCategory.Hom.toOrderHom  …
        -/
      · rw [Fin.castSucc_castPred]
        /-
          🎉 no goals
        -/
        /-
          case pos.h
          n : Nat
          Δ : SimplexCategory
          θ : Quiver.Hom Δ (SimplexCategory.mk (HAdd.hAdd n 1))
          i : Fin (HAdd.hAdd n 2)
          hi : ∀ (x : Fin (HAdd.hAdd Δ.len 1)), Ne ((SimplexCategory.Hom.toOrderHom θ) x …
          h : LT.lt i (Fin.last (HAdd.hAdd n 1))
          x : Fin (HAdd.hAdd Δ.len 1)
          h' : LE.le ((SimplexCategory.Hom.toOrderHom θ) x) i
          ⊢ LT.lt (((SimplexCategory.Hom.toOrderHom θ) x).castPred ⋯).castSucc i
        -/
      · rw [(hi x).le_iff_lt] at h'
        /-
          case pos.h
          n : Nat
          Δ : SimplexCategory
          θ : Quiver.Hom Δ (SimplexCategory.mk (HAdd.hAdd n 1))
          i : Fin (HAdd.hAdd n 2)
          hi : ∀ (x : Fin (HAdd.hAdd Δ.len 1)), Ne ((SimplexCategory.Hom.toOrderHom θ) x …
          h : LT.lt i (Fin.last (HAdd.hAdd n 1))
          x : Fin (HAdd.hAdd Δ.len 1)
          h'✝ : LE.le ((SimplexCategory.Hom.toOrderHom θ) x) i
          h' : LT.lt ((SimplexCategory.Hom.toOrderHom θ) x) i
          ⊢ LT.lt (((SimplexCategory.Hom.toOrderHom θ) x).castPred ⋯).castSucc i
        -/
        exact h'
        /-
          🎉 no goals
        -/
      /-
        case neg
        n : Nat
        Δ : SimplexCategory
        θ : Quiver.Hom Δ (SimplexCategory.mk (HAdd.hAdd n 1))
        i : Fin (HAdd.hAdd n 2)
        hi : ∀ (x : Fin (HAdd.hAdd Δ.len 1)), Ne ((SimplexCategory.Hom.toOrderHom θ) x …
        h : LT.lt i (Fin.last (HAdd.hAdd n 1))
        x : Fin (HAdd.hAdd Δ.len 1)
        h' : Not (LE.le ((SimplexCategory.Hom.toOrderHom θ) x) i)
        ⊢ Eq ((SimplexCategory.Hom.toOrderHom θ) x) ((SimplexCategory.Hom.toOrderHom ( …
      -/
    · simp only [not_le] at h'
      /-
        case neg
        n : Nat
        Δ : SimplexCategory
        θ : Quiver.Hom Δ (SimplexCategory.mk (HAdd.hAdd n 1))
        i : Fin (HAdd.hAdd n 2)
        hi : ∀ (x : Fin (HAdd.hAdd Δ.len 1)), Ne ((SimplexCategory.Hom.toOrderHom θ) x …
        h : LT.lt i (Fin.last (HAdd.hAdd n 1))
        x : Fin (HAdd.hAdd Δ.len 1)
        h' : LT.lt i ((SimplexCategory.Hom.toOrderHom θ) x)
        ⊢ Eq ((SimplexCategory.Hom.toOrderHom θ) x) ((SimplexCategory.Hom.toOrderHom ( …
      -/
      dsimp [σ, δ]
      /-
        case neg
        n : Nat
        Δ : SimplexCategory
        θ : Quiver.Hom Δ (SimplexCategory.mk (HAdd.hAdd n 1))
        i : Fin (HAdd.hAdd n 2)
        hi : ∀ (x : Fin (HAdd.hAdd Δ.len 1)), Ne ((SimplexCategory.Hom.toOrderHom θ) x …
        h : LT.lt i (Fin.last (HAdd.hAdd n 1))
        x : Fin (HAdd.hAdd Δ.len 1)
        h' : LT.lt i ((SimplexCategory.Hom.toOrderHom θ) x)
        ⊢ Eq ((SimplexCategory.Hom.toOrderHom θ) x) (i.succAbove ((i.castPred ⋯).predA …
      -/
      rw [Fin.predAbove_of_castSucc_lt _ _ (by rwa [Fin.castSucc_castPred])]
      /-
        case neg
        n : Nat
        Δ : SimplexCategory
        θ : Quiver.Hom Δ (SimplexCategory.mk (HAdd.hAdd n 1))
        i : Fin (HAdd.hAdd n 2)
        hi : ∀ (x : Fin (HAdd.hAdd Δ.len 1)), Ne ((SimplexCategory.Hom.toOrderHom θ) x …
        h : LT.lt i (Fin.last (HAdd.hAdd n 1))
        x : Fin (HAdd.hAdd Δ.len 1)
        h' : LT.lt i ((SimplexCategory.Hom.toOrderHom θ) x)
        ⊢ Eq ((SimplexCategory.Hom.toOrderHom θ) x) (i.succAbove (((SimplexCategory.Ho …
      -/
      rw [Fin.succAbove_of_le_castSucc i _]
        /-
          case neg
          n : Nat
          Δ : SimplexCategory
          θ : Quiver.Hom Δ (SimplexCategory.mk (HAdd.hAdd n 1))
          i : Fin (HAdd.hAdd n 2)
          hi : ∀ (x : Fin (HAdd.hAdd Δ.len 1)), Ne ((SimplexCategory.Hom.toOrderHom θ) x …
          h : LT.lt i (Fin.last (HAdd.hAdd n 1))
          x : Fin (HAdd.hAdd Δ.len 1)
          h' : LT.lt i ((SimplexCategory.Hom.toOrderHom θ) x)
          ⊢ Eq ((SimplexCategory.Hom.toOrderHom θ) x) (((SimplexCategory.Hom.toOrderHom  …
        -/
      · rw [Fin.succ_pred]
        /-
          🎉 no goals
        -/
        /-
          case neg
          n : Nat
          Δ : SimplexCategory
          θ : Quiver.Hom Δ (SimplexCategory.mk (HAdd.hAdd n 1))
          i : Fin (HAdd.hAdd n 2)
          hi : ∀ (x : Fin (HAdd.hAdd Δ.len 1)), Ne ((SimplexCategory.Hom.toOrderHom θ) x …
          h : LT.lt i (Fin.last (HAdd.hAdd n 1))
          x : Fin (HAdd.hAdd Δ.len 1)
          h' : LT.lt i ((SimplexCategory.Hom.toOrderHom θ) x)
          ⊢ LE.le i (((SimplexCategory.Hom.toOrderHom θ) x).pred ⋯).castSucc
        -/
      · exact Nat.le_sub_one_of_lt (Fin.lt_iff_val_lt_val.mp h')
        /-
          🎉 no goals
        -/
    /-
      case neg
      n : Nat
      Δ : SimplexCategory
      θ : Quiver.Hom Δ (SimplexCategory.mk (HAdd.hAdd n 1))
      i : Fin (HAdd.hAdd n 2)
      hi : ∀ (x : Fin (HAdd.hAdd Δ.len 1)), Ne ((SimplexCategory.Hom.toOrderHom θ) x …
      h : Not (LT.lt i (Fin.last (HAdd.hAdd n 1)))
      ⊢ Exists fun θ' => Eq θ (CategoryTheory.CategoryStruct.comp θ' (SimplexCategor …
    -/
  · obtain rfl := le_antisymm (Fin.le_last i) (not_lt.mp h)
    /-
      case neg
      n : Nat
      Δ : SimplexCategory
      θ : Quiver.Hom Δ (SimplexCategory.mk (HAdd.hAdd n 1))
      hi : ∀ (x : Fin (HAdd.hAdd Δ.len 1)), Ne ((SimplexCategory.Hom.toOrderHom θ) x …
      h : Not (LT.lt (Fin.last (HAdd.hAdd n 1)) (Fin.last (HAdd.hAdd n 1)))
      ⊢ Exists fun θ' => Eq θ (CategoryTheory.CategoryStruct.comp θ' (SimplexCategor …
    -/
    use θ ≫ σ (Fin.last _)
    /-
      case h
      n : Nat
      Δ : SimplexCategory
      θ : Quiver.Hom Δ (SimplexCategory.mk (HAdd.hAdd n 1))
      hi : ∀ (x : Fin (HAdd.hAdd Δ.len 1)), Ne ((SimplexCategory.Hom.toOrderHom θ) x …
      h : Not (LT.lt (Fin.last (HAdd.hAdd n 1)) (Fin.last (HAdd.hAdd n 1)))
      ⊢ Eq θ (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp …
    -/
    ext x : 3
    /-
      case h.a.h.h
      n : Nat
      Δ : SimplexCategory
      θ : Quiver.Hom Δ (SimplexCategory.mk (HAdd.hAdd n 1))
      hi : ∀ (x : Fin (HAdd.hAdd Δ.len 1)), Ne ((SimplexCategory.Hom.toOrderHom θ) x …
      h : Not (LT.lt (Fin.last (HAdd.hAdd n 1)) (Fin.last (HAdd.hAdd n 1)))
      x : Fin (HAdd.hAdd Δ.len 1)
      ⊢ Eq ((SimplexCategory.Hom.toOrderHom θ) x) ((SimplexCategory.Hom.toOrderHom ( …
    -/
    dsimp [δ, σ]
    /-
      case h.a.h.h
      n : Nat
      Δ : SimplexCategory
      θ : Quiver.Hom Δ (SimplexCategory.mk (HAdd.hAdd n 1))
      hi : ∀ (x : Fin (HAdd.hAdd Δ.len 1)), Ne ((SimplexCategory.Hom.toOrderHom θ) x …
      h : Not (LT.lt (Fin.last (HAdd.hAdd n 1)) (Fin.last (HAdd.hAdd n 1)))
      x : Fin (HAdd.hAdd Δ.len 1)
      ⊢ Eq ((SimplexCategory.Hom.toOrderHom θ) x) ((Fin.last (HAdd.hAdd n 1)).succAb …
    -/
    simp_rw [Fin.succAbove_last, Fin.predAbove_last_apply]
    /-
      case h.a.h.h
      n : Nat
      Δ : SimplexCategory
      θ : Quiver.Hom Δ (SimplexCategory.mk (HAdd.hAdd n 1))
      hi : ∀ (x : Fin (HAdd.hAdd Δ.len 1)), Ne ((SimplexCategory.Hom.toOrderHom θ) x …
      h : Not (LT.lt (Fin.last (HAdd.hAdd n 1)) (Fin.last (HAdd.hAdd n 1)))
      x : Fin (HAdd.hAdd Δ.len 1)
      ⊢ Eq ((SimplexCategory.Hom.toOrderHom θ) x) (dite (Eq ((SimplexCategory.Hom.to …
    -/
    erw [dif_neg (hi x)]
    /-
      case h.a.h.h
      n : Nat
      Δ : SimplexCategory
      θ : Quiver.Hom Δ (SimplexCategory.mk (HAdd.hAdd n 1))
      hi : ∀ (x : Fin (HAdd.hAdd Δ.len 1)), Ne ((SimplexCategory.Hom.toOrderHom θ) x …
      h : Not (LT.lt (Fin.last (HAdd.hAdd n 1)) (Fin.last (HAdd.hAdd n 1)))
      x : Fin (HAdd.hAdd Δ.len 1)
      ⊢ Eq ((SimplexCategory.Hom.toOrderHom θ) x) (((SimplexCategory.Hom.toOrderHom  …
    -/
    rw [Fin.castSucc_castPred]
    /-
      🎉 no goals
    -/


theorem eq_comp_δ_of_not_surjective {n : ℕ} {Δ : SimplexCategory} (θ : Δ ⟶ mk (n + 1))
    (hθ : ¬Function.Surjective θ.toOrderHom) :
    ∃ (i : Fin (n + 2)) (θ' : Δ ⟶ mk n), θ = θ' ≫ δ i := by
  /-
    n : Nat
    Δ : SimplexCategory
    θ : Quiver.Hom Δ (SimplexCategory.mk (HAdd.hAdd n 1))
    hθ : Not (Function.Surjective ⇑(SimplexCategory.Hom.toOrderHom θ))
    ⊢ Exists fun i => Exists fun θ' => Eq θ (CategoryTheory.CategoryStruct.comp θ' …
  -/
  cases' not_forall.mp hθ with i hi
  /-
    case intro
    n : Nat
    Δ : SimplexCategory
    θ : Quiver.Hom Δ (SimplexCategory.mk (HAdd.hAdd n 1))
    hθ : Not (Function.Surjective ⇑(SimplexCategory.Hom.toOrderHom θ))
    i : Fin (HAdd.hAdd (SimplexCategory.mk (HAdd.hAdd n 1)).len 1)
    hi : Not (Exists fun a => Eq ((SimplexCategory.Hom.toOrderHom θ) a) i)
    ⊢ Exists fun i => Exists fun θ' => Eq θ (CategoryTheory.CategoryStruct.comp θ' …
  -/
  use i
  /-
    case h
    n : Nat
    Δ : SimplexCategory
    θ : Quiver.Hom Δ (SimplexCategory.mk (HAdd.hAdd n 1))
    hθ : Not (Function.Surjective ⇑(SimplexCategory.Hom.toOrderHom θ))
    i : Fin (HAdd.hAdd (SimplexCategory.mk (HAdd.hAdd n 1)).len 1)
    hi : Not (Exists fun a => Eq ((SimplexCategory.Hom.toOrderHom θ) a) i)
    ⊢ Exists fun θ' => Eq θ (CategoryTheory.CategoryStruct.comp θ' (SimplexCategor …
  -/
  exact eq_comp_δ_of_not_surjective' θ i (not_exists.mp hi)
  /-
    🎉 no goals
  -/


theorem eq_id_of_mono {x : SimplexCategory} (i : x ⟶ x) [Mono i] : i = 𝟙 _ := by
  suffices IsIso i by
    apply eq_id_of_isIso
  /-
    x : SimplexCategory
    i : Quiver.Hom x x
    inst✝ : CategoryTheory.Mono i
    ⊢ CategoryTheory.IsIso i
  -/
  apply isIso_of_bijective
  /-
    case hf
    x : SimplexCategory
    i : Quiver.Hom x x
    inst✝ : CategoryTheory.Mono i
    ⊢ Function.Bijective (SimplexCategory.Hom.toOrderHom i).toFun
  -/
  dsimp
  rw [Fintype.bijective_iff_injective_and_card i.toOrderHom, ← mono_iff_injective,
    eq_self_iff_true, and_true]
  /-
    case hf
    x : SimplexCategory
    i : Quiver.Hom x x
    inst✝ : CategoryTheory.Mono i
    ⊢ CategoryTheory.Mono i
  -/
  infer_instance
  /-
    🎉 no goals
  -/


theorem eq_id_of_epi {x : SimplexCategory} (i : x ⟶ x) [Epi i] : i = 𝟙 _ := by
  suffices IsIso i by
    haveI := this
    apply eq_id_of_isIso
  /-
    x : SimplexCategory
    i : Quiver.Hom x x
    inst✝ : CategoryTheory.Epi i
    ⊢ CategoryTheory.IsIso i
  -/
  apply isIso_of_bijective
  /-
    case hf
    x : SimplexCategory
    i : Quiver.Hom x x
    inst✝ : CategoryTheory.Epi i
    ⊢ Function.Bijective (SimplexCategory.Hom.toOrderHom i).toFun
  -/
  dsimp
  rw [Fintype.bijective_iff_surjective_and_card i.toOrderHom, ← epi_iff_surjective,
    eq_self_iff_true, and_true]
  /-
    case hf
    x : SimplexCategory
    i : Quiver.Hom x x
    inst✝ : CategoryTheory.Epi i
    ⊢ CategoryTheory.Epi i
  -/
  infer_instance
  /-
    🎉 no goals
  -/


theorem eq_σ_of_epi {n : ℕ} (θ : mk (n + 1) ⟶ mk n) [Epi θ] : ∃ i : Fin (n + 1), θ = σ i := by
  rcases eq_σ_comp_of_not_injective θ (by
    by_contra h
    simpa using le_of_mono (mono_iff_injective.mpr h)) with ⟨i, θ', h⟩
  /-
    case intro.intro
    n : Nat
    θ : Quiver.Hom (SimplexCategory.mk (HAdd.hAdd n 1)) (SimplexCategory.mk n)
    inst✝ : CategoryTheory.Epi θ
    i : Fin (HAdd.hAdd n 1)
    θ' : Quiver.Hom (SimplexCategory.mk n) (SimplexCategory.mk n)
    h : Eq θ (CategoryTheory.CategoryStruct.comp (SimplexCategory.σ i) θ')
    ⊢ Exists fun i => Eq θ (SimplexCategory.σ i)
  -/
  use i
  haveI : Epi (σ i ≫ θ') := by
    rw [← h]
    infer_instance
  /-
    case h
    n : Nat
    θ : Quiver.Hom (SimplexCategory.mk (HAdd.hAdd n 1)) (SimplexCategory.mk n)
    inst✝ : CategoryTheory.Epi θ
    i : Fin (HAdd.hAdd n 1)
    θ' : Quiver.Hom (SimplexCategory.mk n) (SimplexCategory.mk n)
    h : Eq θ (CategoryTheory.CategoryStruct.comp (SimplexCategory.σ i) θ')
    this : CategoryTheory.Epi (CategoryTheory.CategoryStruct.comp (SimplexCategory …
    ⊢ Eq θ (SimplexCategory.σ i)
  -/
  haveI := CategoryTheory.epi_of_epi (σ i) θ'
  /-
    case h
    n : Nat
    θ : Quiver.Hom (SimplexCategory.mk (HAdd.hAdd n 1)) (SimplexCategory.mk n)
    inst✝ : CategoryTheory.Epi θ
    i : Fin (HAdd.hAdd n 1)
    θ' : Quiver.Hom (SimplexCategory.mk n) (SimplexCategory.mk n)
    h : Eq θ (CategoryTheory.CategoryStruct.comp (SimplexCategory.σ i) θ')
    this✝ : CategoryTheory.Epi (CategoryTheory.CategoryStruct.comp (SimplexCategor …
    this : CategoryTheory.Epi θ'
    ⊢ Eq θ (SimplexCategory.σ i)
  -/
  rw [h, eq_id_of_epi θ', Category.comp_id]
  /-
    🎉 no goals
  -/


theorem eq_δ_of_mono {n : ℕ} (θ : mk n ⟶ mk (n + 1)) [Mono θ] : ∃ i : Fin (n + 2), θ = δ i := by
  rcases eq_comp_δ_of_not_surjective θ (by
    by_contra h
    simpa using le_of_epi (epi_iff_surjective.mpr h)) with ⟨i, θ', h⟩
  /-
    case intro.intro
    n : Nat
    θ : Quiver.Hom (SimplexCategory.mk n) (SimplexCategory.mk (HAdd.hAdd n 1))
    inst✝ : CategoryTheory.Mono θ
    i : Fin (HAdd.hAdd n 2)
    θ' : Quiver.Hom (SimplexCategory.mk n) (SimplexCategory.mk n)
    h : Eq θ (CategoryTheory.CategoryStruct.comp θ' (SimplexCategory.δ i))
    ⊢ Exists fun i => Eq θ (SimplexCategory.δ i)
  -/
  use i
  haveI : Mono (θ' ≫ δ i) := by
    rw [← h]
    infer_instance
  /-
    case h
    n : Nat
    θ : Quiver.Hom (SimplexCategory.mk n) (SimplexCategory.mk (HAdd.hAdd n 1))
    inst✝ : CategoryTheory.Mono θ
    i : Fin (HAdd.hAdd n 2)
    θ' : Quiver.Hom (SimplexCategory.mk n) (SimplexCategory.mk n)
    h : Eq θ (CategoryTheory.CategoryStruct.comp θ' (SimplexCategory.δ i))
    this : CategoryTheory.Mono (CategoryTheory.CategoryStruct.comp θ' (SimplexCate …
    ⊢ Eq θ (SimplexCategory.δ i)
  -/
  haveI := CategoryTheory.mono_of_mono θ' (δ i)
  /-
    case h
    n : Nat
    θ : Quiver.Hom (SimplexCategory.mk n) (SimplexCategory.mk (HAdd.hAdd n 1))
    inst✝ : CategoryTheory.Mono θ
    i : Fin (HAdd.hAdd n 2)
    θ' : Quiver.Hom (SimplexCategory.mk n) (SimplexCategory.mk n)
    h : Eq θ (CategoryTheory.CategoryStruct.comp θ' (SimplexCategory.δ i))
    this✝ : CategoryTheory.Mono (CategoryTheory.CategoryStruct.comp θ' (SimplexCat …
    this : CategoryTheory.Mono θ'
    ⊢ Eq θ (SimplexCategory.δ i)
  -/
  rw [h, eq_id_of_mono θ', Category.id_comp]
  /-
    🎉 no goals
  -/


theorem len_lt_of_mono {Δ' Δ : SimplexCategory} (i : Δ' ⟶ Δ) [hi : Mono i] (hi' : Δ ≠ Δ') :
    Δ'.len < Δ.len := by
  /-
    Δ' Δ : SimplexCategory
    i : Quiver.Hom Δ' Δ
    hi : CategoryTheory.Mono i
    hi' : Ne Δ Δ'
    ⊢ LT.lt Δ'.len Δ.len
  -/
  rcases lt_or_eq_of_le (len_le_of_mono hi) with (h | h)
    /-
      case inl
      Δ' Δ : SimplexCategory
      i : Quiver.Hom Δ' Δ
      hi : CategoryTheory.Mono i
      hi' : Ne Δ Δ'
      h : LT.lt Δ'.len Δ.len
      ⊢ LT.lt Δ'.len Δ.len
    -/
  · exact h
    /-
      🎉 no goals
    -/
    /-
      case inr
      Δ' Δ : SimplexCategory
      i : Quiver.Hom Δ' Δ
      hi : CategoryTheory.Mono i
      hi' : Ne Δ Δ'
      h : Eq Δ'.len Δ.len
      ⊢ LT.lt Δ'.len Δ.len
    -/
  · exfalso
    /-
      case inr
      Δ' Δ : SimplexCategory
      i : Quiver.Hom Δ' Δ
      hi : CategoryTheory.Mono i
      hi' : Ne Δ Δ'
      h : Eq Δ'.len Δ.len
      ⊢ False
    -/
    exact hi' (by ext; exact h.symm)
    /-
      🎉 no goals
    -/


noncomputable instance : SplitEpiCategory SimplexCategory :=
  skeletalEquivalence.inverse.splitEpiCategoryImpOfIsEquivalence


instance : HasStrongEpiMonoFactorisations SimplexCategory :=
  Functor.hasStrongEpiMonoFactorisations_imp_of_isEquivalence
    SimplexCategory.skeletalEquivalence.inverse


instance : HasStrongEpiImages SimplexCategory :=
  Limits.hasStrongEpiImages_of_hasStrongEpiMonoFactorisations


instance (Δ Δ' : SimplexCategory) (θ : Δ ⟶ Δ') : Epi (factorThruImage θ) :=
  StrongEpi.epi


theorem image_eq {Δ Δ' Δ'' : SimplexCategory} {φ : Δ ⟶ Δ''} {e : Δ ⟶ Δ'} [Epi e] {i : Δ' ⟶ Δ''}
    [Mono i] (fac : e ≫ i = φ) : image φ = Δ' := by
  /-
    Δ Δ' Δ'' : SimplexCategory
    φ : Quiver.Hom Δ Δ''
    e : Quiver.Hom Δ Δ'
    inst✝¹ : CategoryTheory.Epi e
    i : Quiver.Hom Δ' Δ''
    inst✝ : CategoryTheory.Mono i
    fac : Eq (CategoryTheory.CategoryStruct.comp e i) φ
    ⊢ Eq (CategoryTheory.Limits.image φ) Δ'
  -/
  haveI := strongEpi_of_epi e
  /-
    Δ Δ' Δ'' : SimplexCategory
    φ : Quiver.Hom Δ Δ''
    e : Quiver.Hom Δ Δ'
    inst✝¹ : CategoryTheory.Epi e
    i : Quiver.Hom Δ' Δ''
    inst✝ : CategoryTheory.Mono i
    fac : Eq (CategoryTheory.CategoryStruct.comp e i) φ
    this : CategoryTheory.StrongEpi e
    ⊢ Eq (CategoryTheory.Limits.image φ) Δ'
  -/
  let e := image.isoStrongEpiMono e i fac
  /-
    Δ Δ' Δ'' : SimplexCategory
    φ : Quiver.Hom Δ Δ''
    e✝ : Quiver.Hom Δ Δ'
    inst✝¹ : CategoryTheory.Epi e✝
    i : Quiver.Hom Δ' Δ''
    inst✝ : CategoryTheory.Mono i
    fac : Eq (CategoryTheory.CategoryStruct.comp e✝ i) φ
    this : CategoryTheory.StrongEpi e✝
    e : CategoryTheory.Iso Δ' (CategoryTheory.Limits.image φ) := CategoryTheory.Li …
    ⊢ Eq (CategoryTheory.Limits.image φ) Δ'
  -/
  ext
  exact
    le_antisymm (len_le_of_epi (inferInstance : Epi e.hom))
      (len_le_of_mono (inferInstance : Mono e.hom))


theorem image_ι_eq {Δ Δ'' : SimplexCategory} {φ : Δ ⟶ Δ''} {e : Δ ⟶ image φ} [Epi e]
    {i : image φ ⟶ Δ''} [Mono i] (fac : e ≫ i = φ) : image.ι φ = i := by
  /-
    Δ Δ'' : SimplexCategory
    φ : Quiver.Hom Δ Δ''
    e : Quiver.Hom Δ (CategoryTheory.Limits.image φ)
    inst✝¹ : CategoryTheory.Epi e
    i : Quiver.Hom (CategoryTheory.Limits.image φ) Δ''
    inst✝ : CategoryTheory.Mono i
    fac : Eq (CategoryTheory.CategoryStruct.comp e i) φ
    ⊢ Eq (CategoryTheory.Limits.image.ι φ) i
  -/
  haveI := strongEpi_of_epi e
  rw [← image.isoStrongEpiMono_hom_comp_ι e i fac,
    SimplexCategory.eq_id_of_isIso (image.isoStrongEpiMono e i fac).hom, Category.id_comp]


theorem factorThruImage_eq {Δ Δ'' : SimplexCategory} {φ : Δ ⟶ Δ''} {e : Δ ⟶ image φ} [Epi e]
    {i : image φ ⟶ Δ''} [Mono i] (fac : e ≫ i = φ) : factorThruImage φ = e := by
  /-
    Δ Δ'' : SimplexCategory
    φ : Quiver.Hom Δ Δ''
    e : Quiver.Hom Δ (CategoryTheory.Limits.image φ)
    inst✝¹ : CategoryTheory.Epi e
    i : Quiver.Hom (CategoryTheory.Limits.image φ) Δ''
    inst✝ : CategoryTheory.Mono i
    fac : Eq (CategoryTheory.CategoryStruct.comp e i) φ
    ⊢ Eq (CategoryTheory.Limits.factorThruImage φ) e
  -/
  rw [← cancel_mono i, fac, ← image_ι_eq fac, image.fac]
  /-
    🎉 no goals
  -/


/-- This functor `SimplexCategory ⥤ Cat` sends `[n]` (for `n : ℕ`)
to the category attached to the ordered set `{0, 1, ..., n}` -/
@[simps! obj map]
def toCat : SimplexCategory ⥤ Cat.{0} :=
  SimplexCategory.skeletalFunctor ⋙ forget₂ NonemptyFinLinOrd LinOrd ⋙
      forget₂ LinOrd Lat ⋙ forget₂ Lat PartOrd ⋙
      forget₂ PartOrd Preord ⋙ preordToCat


