/-- Take the first `m` elements of an `n`-tuple where `m ≤ n`, returning an `m`-tuple. -/
def take (m : ℕ) (h : m ≤ n) (v : (i : Fin n) → α i) : (i : Fin m) → α (castLE h i) :=
  fun i ↦ v (castLE h i)


@[simp]
theorem take_apply (m : ℕ) (h : m ≤ n) (v : (i : Fin n) → α i) (i : Fin m) :
    (take m h v) i = v (castLE h i) := rfl


@[simp]
theorem take_zero (v : (i : Fin n) → α i) : take 0 n.zero_le v = fun i ↦ elim0 i := by
  /-
    n : Nat
    α : Fin n → Sort u_1
    v : (i : Fin n) → α i
    ⊢ Eq (Fin.take 0 ⋯ v) fun i => i.elim0
  -/
  ext i; exact elim0 i
         /-
           🎉 no goals
         -/


@[simp]
theorem take_one {α : Fin (n + 1) → Sort*} (v : (i : Fin (n + 1)) → α i) :
    take 1 (Nat.le_add_left 1 n) v = (fun i => v (castLE (Nat.le_add_left 1 n) i)) := by
  /-
    n : Nat
    α : Fin (HAdd.hAdd n 1) → Sort u_2
    v : (i : Fin (HAdd.hAdd n 1)) → α i
    ⊢ Eq (Fin.take 1 ⋯ v) fun i => v (Fin.castLE ⋯ i)
  -/
  ext i
  /-
    case h
    n : Nat
    α : Fin (HAdd.hAdd n 1) → Sort u_2
    v : (i : Fin (HAdd.hAdd n 1)) → α i
    i : Fin 1
    ⊢ Eq (Fin.take 1 ⋯ v i) (v (Fin.castLE ⋯ i))
  -/
  simp only [take]
  /-
    🎉 no goals
  -/


@[simp]
theorem take_eq_init {α : Fin (n + 1) → Sort*} (v : (i : Fin (n + 1)) → α i) :
    take n n.le_succ v = init v := by
  /-
    n : Nat
    α : Fin (HAdd.hAdd n 1) → Sort u_2
    v : (i : Fin (HAdd.hAdd n 1)) → α i
    ⊢ Eq (Fin.take n ⋯ v) (Fin.init v)
  -/
  ext i
  /-
    case h
    n : Nat
    α : Fin (HAdd.hAdd n 1) → Sort u_2
    v : (i : Fin (HAdd.hAdd n 1)) → α i
    i : Fin n
    ⊢ Eq (Fin.take n ⋯ v i) (Fin.init v i)
  -/
  simp only [Nat.succ_eq_add_one, take, init]
  /-
    case h
    n : Nat
    α : Fin (HAdd.hAdd n 1) → Sort u_2
    v : (i : Fin (HAdd.hAdd n 1)) → α i
    i : Fin n
    ⊢ Eq (v (Fin.castLE ⋯ i)) (v i.castSucc)
  -/
  congr
  /-
    🎉 no goals
  -/


@[simp]
theorem take_eq_self (v : (i : Fin n) → α i) : take n (le_refl n) v = v := by
  /-
    n : Nat
    α : Fin n → Sort u_1
    v : (i : Fin n) → α i
    ⊢ Eq (Fin.take n ⋯ v) v
  -/
  ext i
  /-
    case h
    n : Nat
    α : Fin n → Sort u_1
    v : (i : Fin n) → α i
    i : Fin n
    ⊢ Eq (Fin.take n ⋯ v i) (v i)
  -/
  simp [take]
  /-
    🎉 no goals
  -/


@[simp]
theorem take_take {m n' : ℕ} (h : m ≤ n') (h' : n' ≤ n) (v : (i : Fin n) → α i) :
    take m h (take n' h' v) = take m (Nat.le_trans h h') v := by
  /-
    n : Nat
    α : Fin n → Sort u_1
    m n' : Nat
    h : LE.le m n'
    h' : LE.le n' n
    v : (i : Fin n) → α i
    ⊢ Eq (Fin.take m h (Fin.take n' h' v)) (Fin.take m ⋯ v)
  -/
  ext i
  /-
    case h
    n : Nat
    α : Fin n → Sort u_1
    m n' : Nat
    h : LE.le m n'
    h' : LE.le n' n
    v : (i : Fin n) → α i
    i : Fin m
    ⊢ Eq (Fin.take m h (Fin.take n' h' v) i) (Fin.take m ⋯ v i)
  -/
  simp only [take]
  /-
    case h
    n : Nat
    α : Fin n → Sort u_1
    m n' : Nat
    h : LE.le m n'
    h' : LE.le n' n
    v : (i : Fin n) → α i
    i : Fin m
    ⊢ Eq (v (Fin.castLE h' (Fin.castLE h i))) (v (Fin.castLE ⋯ i))
  -/
  congr
  /-
    🎉 no goals
  -/


@[simp]
theorem take_init {α : Fin (n + 1) → Sort*} (m : ℕ) (h : m ≤ n) (v : (i : Fin (n + 1)) → α i) :
    take m h (init v) = take m (Nat.le_succ_of_le h) v := by
  /-
    n : Nat
    α : Fin (HAdd.hAdd n 1) → Sort u_2
    m : Nat
    h : LE.le m n
    v : (i : Fin (HAdd.hAdd n 1)) → α i
    ⊢ Eq (Fin.take m h (Fin.init v)) (Fin.take m ⋯ v)
  -/
  ext i
  /-
    case h
    n : Nat
    α : Fin (HAdd.hAdd n 1) → Sort u_2
    m : Nat
    h : LE.le m n
    v : (i : Fin (HAdd.hAdd n 1)) → α i
    i : Fin m
    ⊢ Eq (Fin.take m h (Fin.init v) i) (Fin.take m ⋯ v i)
  -/
  simp only [take, init]
  /-
    case h
    n : Nat
    α : Fin (HAdd.hAdd n 1) → Sort u_2
    m : Nat
    h : LE.le m n
    v : (i : Fin (HAdd.hAdd n 1)) → α i
    i : Fin m
    ⊢ Eq (v (Fin.castLE h i).castSucc) (v (Fin.castLE ⋯ i))
  -/
  congr
  /-
    🎉 no goals
  -/


theorem take_repeat {α : Type*} {n' : ℕ} (m : ℕ) (h : m ≤ n) (a : Fin n' → α) :
    take (m * n') (Nat.mul_le_mul_right n' h) (Fin.repeat n a) = Fin.repeat m a := by
  /-
    n : Nat
    α : Type u_2
    n' m : Nat
    h : LE.le m n
    a : Fin n' → α
    ⊢ Eq (Fin.take (HMul.hMul m n') ⋯ (Fin.repeat n a)) (Fin.repeat m a)
  -/
  ext i
  /-
    case h
    n : Nat
    α : Type u_2
    n' m : Nat
    h : LE.le m n
    a : Fin n' → α
    i : Fin (HMul.hMul m n')
    ⊢ Eq (Fin.take (HMul.hMul m n') ⋯ (Fin.repeat n a) i) (Fin.repeat m a i)
  -/
  simp only [take, repeat_apply, modNat, coe_castLE]
  /-
    🎉 no goals
  -/


/-- Taking `m + 1` elements is equal to taking `m` elements and adding the `(m + 1)`th one. -/
theorem take_succ_eq_snoc (m : ℕ) (h : m < n) (v : (i : Fin n) → α i) :
    take m.succ h v = snoc (take m h.le v) (v ⟨m, h⟩) := by
  /-
    n : Nat
    α : Fin n → Sort u_1
    m : Nat
    h : LT.lt m n
    v : (i : Fin n) → α i
    ⊢ Eq (Fin.take m.succ h v) (Fin.snoc (Fin.take m ⋯ v) (v ⟨m, h⟩))
  -/
  ext i
  induction m with
  | zero =>
    have h' : i = 0 := by ext; simp
    subst h'
    simp [take, snoc, castLE]
  | succ m _ =>
    induction i using reverseInduction with
    | last => simp [take, snoc, castLT]; congr
    | cast i _ => simp [snoc_cast_add]


/-- `take` commutes with `update` for indices in the range of `take`. -/
@[simp]
theorem take_update_of_lt (m : ℕ) (h : m ≤ n) (v : (i : Fin n) → α i) (i : Fin m)
    (x : α (castLE h i)) : take m h (update v (castLE h i) x) = update (take m h v) i x := by
  /-
    n : Nat
    α : Fin n → Sort u_1
    m : Nat
    h : LE.le m n
    v : (i : Fin n) → α i
    i : Fin m
    x : α (Fin.castLE h i)
    ⊢ Eq (Fin.take m h (Function.update v (Fin.castLE h i) x)) (Function.update (F …
  -/
  ext j
  /-
    case h
    n : Nat
    α : Fin n → Sort u_1
    m : Nat
    h : LE.le m n
    v : (i : Fin n) → α i
    i : Fin m
    x : α (Fin.castLE h i)
    j : Fin m
    ⊢ Eq (Fin.take m h (Function.update v (Fin.castLE h i) x) j) (Function.update  …
  -/
  by_cases h' : j = i
    /-
      case pos
      n : Nat
      α : Fin n → Sort u_1
      m : Nat
      h : LE.le m n
      v : (i : Fin n) → α i
      i : Fin m
      x : α (Fin.castLE h i)
      j : Fin m
      h' : Eq j i
      ⊢ Eq (Fin.take m h (Function.update v (Fin.castLE h i) x) j) (Function.update  …
    -/
  · rw [h']
    /-
      case pos
      n : Nat
      α : Fin n → Sort u_1
      m : Nat
      h : LE.le m n
      v : (i : Fin n) → α i
      i : Fin m
      x : α (Fin.castLE h i)
      j : Fin m
      h' : Eq j i
      ⊢ Eq (Fin.take m h (Function.update v (Fin.castLE h i) x) i) (Function.update  …
    -/
    simp only [take, update_self]
    /-
      🎉 no goals
    -/
    /-
      case neg
      n : Nat
      α : Fin n → Sort u_1
      m : Nat
      h : LE.le m n
      v : (i : Fin n) → α i
      i : Fin m
      x : α (Fin.castLE h i)
      j : Fin m
      h' : Not (Eq j i)
      ⊢ Eq (Fin.take m h (Function.update v (Fin.castLE h i) x) j) (Function.update  …
    -/
  · have : castLE h j ≠ castLE h i := by simp [h']
    /-
      case neg
      n : Nat
      α : Fin n → Sort u_1
      m : Nat
      h : LE.le m n
      v : (i : Fin n) → α i
      i : Fin m
      x : α (Fin.castLE h i)
      j : Fin m
      h' : Not (Eq j i)
      this : Ne (Fin.castLE h j) (Fin.castLE h i)
      ⊢ Eq (Fin.take m h (Function.update v (Fin.castLE h i) x) j) (Function.update  …
    -/
    simp only [take, update_of_ne h', update_of_ne this]
    /-
      🎉 no goals
    -/


/-- `take` is the same after `update` for indices outside the range of `take`. -/
@[simp]
theorem take_update_of_ge (m : ℕ) (h : m ≤ n) (v : (i : Fin n) → α i) (i : Fin n) (hi : i ≥ m)
    (x : α i) : take m h (update v i x) = take m h v := by
  /-
    n : Nat
    α : Fin n → Sort u_1
    m : Nat
    h : LE.le m n
    v : (i : Fin n) → α i
    i : Fin n
    hi : GE.ge (↑i) m
    x : α i
    ⊢ Eq (Fin.take m h (Function.update v i x)) (Fin.take m h v)
  -/
  ext j
  have : castLE h j ≠ i := by
    refine ne_of_val_ne ?_
    simp only [coe_castLE]
    exact Nat.ne_of_lt (lt_of_lt_of_le j.isLt hi)
  /-
    case h
    n : Nat
    α : Fin n → Sort u_1
    m : Nat
    h : LE.le m n
    v : (i : Fin n) → α i
    i : Fin n
    hi : GE.ge (↑i) m
    x : α i
    j : Fin m
    this : Ne (Fin.castLE h j) i
    ⊢ Eq (Fin.take m h (Function.update v i x) j) (Fin.take m h v j)
  -/
  simp only [take, update_of_ne this]
  /-
    🎉 no goals
  -/


/-- Taking the first `m ≤ n` elements of an `addCases u v`, where `u` is a `n`-tuple, is the same as
taking the first `m` elements of `u`. -/
theorem take_addCases_left {n' : ℕ} {motive : Fin (n + n') → Sort*} (m : ℕ) (h : m ≤ n)
    (u : (i : Fin n) → motive (castAdd n' i)) (v : (i : Fin n') → motive (natAdd n i)) :
      take m (Nat.le_add_right_of_le h) (addCases u v) = take m h u := by
  /-
    n n' : Nat
    motive : Fin (HAdd.hAdd n n') → Sort u_2
    m : Nat
    h : LE.le m n
    u : (i : Fin n) → motive (Fin.castAdd n' i)
    v : (i : Fin n') → motive (Fin.natAdd n i)
    ⊢ Eq (Fin.take m ⋯ fun i => Fin.addCases u v i) (Fin.take m h u)
  -/
  ext i
  /-
    case h
    n n' : Nat
    motive : Fin (HAdd.hAdd n n') → Sort u_2
    m : Nat
    h : LE.le m n
    u : (i : Fin n) → motive (Fin.castAdd n' i)
    v : (i : Fin n') → motive (Fin.natAdd n i)
    i : Fin m
    ⊢ Eq (Fin.take m ⋯ (fun i => Fin.addCases u v i) i) (Fin.take m h u i)
  -/
  have : i < n := Nat.lt_of_lt_of_le i.isLt h
  /-
    case h
    n n' : Nat
    motive : Fin (HAdd.hAdd n n') → Sort u_2
    m : Nat
    h : LE.le m n
    u : (i : Fin n) → motive (Fin.castAdd n' i)
    v : (i : Fin n') → motive (Fin.natAdd n i)
    i : Fin m
    this : LT.lt (↑i) n
    ⊢ Eq (Fin.take m ⋯ (fun i => Fin.addCases u v i) i) (Fin.take m h u i)
  -/
  simp only [take, addCases, this, coe_castLE, ↓reduceDIte]
  /-
    case h
    n n' : Nat
    motive : Fin (HAdd.hAdd n n') → Sort u_2
    m : Nat
    h : LE.le m n
    u : (i : Fin n) → motive (Fin.castAdd n' i)
    v : (i : Fin n') → motive (Fin.natAdd n i)
    i : Fin m
    this : LT.lt (↑i) n
    ⊢ Eq (u ((Fin.castLE ⋯ i).castLT ⋯)) (u (Fin.castLE h i))
  -/
  congr
  /-
    🎉 no goals
  -/


/-- Version of `take_addCases_left` that specializes `addCases` to `append`. -/
theorem take_append_left {n' : ℕ} {α : Sort*} (m : ℕ) (h : m ≤ n) (u : (i : Fin n) → α)
    (v : (i : Fin n') → α) : take m (Nat.le_add_right_of_le h) (append u v) = take m h u :=
  take_addCases_left m h _ _


/-- Taking the first `n + m` elements of an `addCases u v`, where `v` is a `n'`-tuple and `m ≤ n'`,
is the same as appending `u` with the first `m` elements of `v`. -/
theorem take_addCases_right {n' : ℕ} {motive : Fin (n + n') → Sort*} (m : ℕ) (h : m ≤ n')
    (u : (i : Fin n) → motive (castAdd n' i)) (v : (i : Fin n') → motive (natAdd n i)) :
      take (n + m) (Nat.add_le_add_left h n) (addCases u v) = addCases u (take m h v) := by
  /-
    n n' : Nat
    motive : Fin (HAdd.hAdd n n') → Sort u_2
    m : Nat
    h : LE.le m n'
    u : (i : Fin n) → motive (Fin.castAdd n' i)
    v : (i : Fin n') → motive (Fin.natAdd n i)
    ⊢ Eq (Fin.take (HAdd.hAdd n m) ⋯ fun i => Fin.addCases u v i) fun i => Fin.add …
  -/
  ext i
  /-
    case h
    n n' : Nat
    motive : Fin (HAdd.hAdd n n') → Sort u_2
    m : Nat
    h : LE.le m n'
    u : (i : Fin n) → motive (Fin.castAdd n' i)
    v : (i : Fin n') → motive (Fin.natAdd n i)
    i : Fin (HAdd.hAdd n m)
    ⊢ Eq (Fin.take (HAdd.hAdd n m) ⋯ (fun i => Fin.addCases u v i) i) (Fin.addCase …
  -/
  simp only [take, addCases, coe_castLE]
  /-
    case h
    n n' : Nat
    motive : Fin (HAdd.hAdd n n') → Sort u_2
    m : Nat
    h : LE.le m n'
    u : (i : Fin n) → motive (Fin.castAdd n' i)
    v : (i : Fin n') → motive (Fin.natAdd n i)
    i : Fin (HAdd.hAdd n m)
    ⊢ Eq (dite (LT.lt (↑i) n) (fun h_1 => u ((Fin.castLE ⋯ i).castLT ⋯)) fun h_1 = …
  -/
  by_cases h' : i < n
    /-
      case pos
      n n' : Nat
      motive : Fin (HAdd.hAdd n n') → Sort u_2
      m : Nat
      h : LE.le m n'
      u : (i : Fin n) → motive (Fin.castAdd n' i)
      v : (i : Fin n') → motive (Fin.natAdd n i)
      i : Fin (HAdd.hAdd n m)
      h' : LT.lt (↑i) n
      ⊢ Eq (dite (LT.lt (↑i) n) (fun h_1 => u ((Fin.castLE ⋯ i).castLT ⋯)) fun h_1 = …
    -/
  · simp only [h', ↓reduceDIte]
    /-
      case pos
      n n' : Nat
      motive : Fin (HAdd.hAdd n n') → Sort u_2
      m : Nat
      h : LE.le m n'
      u : (i : Fin n) → motive (Fin.castAdd n' i)
      v : (i : Fin n') → motive (Fin.natAdd n i)
      i : Fin (HAdd.hAdd n m)
      h' : LT.lt (↑i) n
      ⊢ Eq (u ((Fin.castLE ⋯ i).castLT ⋯)) (u (i.castLT ⋯))
    -/
    congr
    /-
      🎉 no goals
    -/
    /-
      case neg
      n n' : Nat
      motive : Fin (HAdd.hAdd n n') → Sort u_2
      m : Nat
      h : LE.le m n'
      u : (i : Fin n) → motive (Fin.castAdd n' i)
      v : (i : Fin n') → motive (Fin.natAdd n i)
      i : Fin (HAdd.hAdd n m)
      h' : Not (LT.lt (↑i) n)
      ⊢ Eq (dite (LT.lt (↑i) n) (fun h_1 => u ((Fin.castLE ⋯ i).castLT ⋯)) fun h_1 = …
    -/
  · simp only [h', ↓reduceDIte, subNat, castLE, cast, eqRec_eq_cast]
    /-
      🎉 no goals
    -/


/-- Version of `take_addCases_right` that specializes `addCases` to `append`. -/
theorem take_append_right {n' : ℕ} {α : Sort*} (m : ℕ) (h : m ≤ n') (u : (i : Fin n) → α)
    (v : (i : Fin n') → α) : take (n + m) (Nat.add_le_add_left h n) (append u v)
        = append u (take m h v) :=
  take_addCases_right m h _ _


/-- `Fin.take` intertwines with `List.take` via `List.ofFn`. -/
theorem ofFn_take_eq_take_ofFn {α : Type*} {m : ℕ} (h : m ≤ n) (v : Fin n → α) :
    List.ofFn (take m h v) = (List.ofFn v).take m :=
                   /-
                     n : Nat
                     α : Type u_2
                     m : Nat
                     h : LE.le m n
                     v : Fin n → α
                     ⊢ Eq (List.ofFn (Fin.take m h v)).length (List.take m (List.ofFn v)).length
                   -/
                   /-
                     🎉 no goals
                   -/
  List.ext_get (by simp [h]) (fun n h1 h2 => by simp)
                                                /-
                                                  🎉 no goals
                                                -/


/-- Alternative version of `take_eq_take_list_ofFn` with `l : List α` instead of `v : Fin n → α`. -/
theorem ofFn_take_get {α : Type*} {m : ℕ} (l : List α) (h : m ≤ l.length) :
    List.ofFn (take m h l.get) = l.take m :=
                   /-
                     α : Type u_2
                     m : Nat
                     l : List α
                     h : LE.le m l.length
                     ⊢ Eq (List.ofFn (Fin.take m h l.get)).length (List.take m l).length
                   -/
                   /-
                     🎉 no goals
                   -/
  List.ext_get (by simp [h]) (fun n h1 h2 => by simp)
                                                /-
                                                  🎉 no goals
                                                -/


/-- `Fin.take` intertwines with `List.take` via `List.get`. -/
theorem get_take_eq_take_get_comp_cast {α : Type*} {m : ℕ} (l : List α) (h : m ≤ l.length) :
    (l.take m).get = take m h l.get ∘ Fin.cast (List.length_take_of_le h) := by
  /-
    α : Type u_2
    m : Nat
    l : List α
    h : LE.le m l.length
    ⊢ Eq (List.take m l).get (Function.comp (Fin.take m h l.get) (Fin.cast ⋯))
  -/
  ext i
  /-
    case h
    α : Type u_2
    m : Nat
    l : List α
    h : LE.le m l.length
    i : Fin (List.take m l).length
    ⊢ Eq ((List.take m l).get i) (Function.comp (Fin.take m h l.get) (Fin.cast ⋯) i)
  -/
  simp only [List.get_eq_getElem, List.getElem_take, comp_apply, take_apply, coe_castLE, coe_cast]
  /-
    🎉 no goals
  -/


/-- Alternative version of `take_eq_take_list_get` with `v : Fin n → α` instead of `l : List α`. -/
theorem get_take_ofFn_eq_take_comp_cast {α : Type*} {m : ℕ} (v : Fin n → α) (h : m ≤ n) :
                                                           /-
                                                             n : Nat
                                                             α✝ : Fin n → Sort u_1
                                                             α : Type u_2
                                                             m : Nat
                                                             v : Fin n → α
                                                             h : LE.le m n
                                                             ⊢ Eq (List.take m (List.ofFn v)).length m
                                                           -/
    ((List.ofFn v).take m).get = take m h v ∘ Fin.cast (by simp [h]) := by
                                                           /-
                                                             🎉 no goals
                                                           -/
  /-
    n : Nat
    α : Type u_2
    m : Nat
    v : Fin n → α
    h : LE.le m n
    ⊢ Eq (List.take m (List.ofFn v)).get (Function.comp (Fin.take m h v) (Fin.cast …
  -/
  ext i
  /-
    case h
    n : Nat
    α : Type u_2
    m : Nat
    v : Fin n → α
    h : LE.le m n
    i : Fin (List.take m (List.ofFn v)).length
    ⊢ Eq ((List.take m (List.ofFn v)).get i) (Function.comp (Fin.take m h v) (Fin. …
  -/
  simp [castLE]
  /-
    🎉 no goals
  -/


