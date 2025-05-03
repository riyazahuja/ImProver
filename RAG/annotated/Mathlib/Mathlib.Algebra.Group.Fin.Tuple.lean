@[to_additive (attr := simp)]
lemma insertNth_one_right [∀ j, One (α j)] (i : Fin (n + 1)) (x : α i) :
    i.insertNth x 1 = Pi.mulSingle i x :=
                           /-
                             n : Nat
                             α : Fin (HAdd.hAdd n 1) → Type u_1
                             inst✝ : (j : Fin (HAdd.hAdd n 1)) → One (α j)
                             i : Fin (HAdd.hAdd n 1)
                             x : α i
                             ⊢ And (Eq x (Pi.mulSingle i x i)) (Eq 1 (i.removeNth (Pi.mulSingle i x)))
                           -/
  insertNth_eq_iff.2 <| by unfold removeNth; simp [succAbove_ne, Pi.one_def]
                                             /-
                                               🎉 no goals
                                             -/


@[to_additive (attr := simp)]
lemma insertNth_mul [∀ j, Mul (α j)] (i : Fin (n + 1)) (x y : α i) (p q : ∀ j, α (i.succAbove j)) :
    i.insertNth (x * y) (p * q) = i.insertNth x p * i.insertNth y q :=
  insertNth_binop (fun _ ↦ (· * ·)) i x y p q


@[to_additive (attr := simp)]
lemma insertNth_div [∀ j, Div (α j)] (i : Fin (n + 1)) (x y : α i)(p q : ∀ j, α (i.succAbove j)) :
    i.insertNth (x / y) (p / q) = i.insertNth x p / i.insertNth y q :=
  insertNth_binop (fun _ ↦ (· / ·)) i x y p q


@[to_additive (attr := simp)]
lemma insertNth_div_same [∀ j, Group (α j)] (i : Fin (n + 1)) (x y : α i)
    (p : ∀ j, α (i.succAbove j)) : i.insertNth x p / i.insertNth y p = Pi.mulSingle i (x / y) := by
  /-
    n : Nat
    α : Fin (HAdd.hAdd n 1) → Type u_1
    inst✝ : (j : Fin (HAdd.hAdd n 1)) → Group (α j)
    i : Fin (HAdd.hAdd n 1)
    x y : α i
    p : (j : Fin n) → α (i.succAbove j)
    ⊢ Eq (HDiv.hDiv (i.insertNth x p) (i.insertNth y p)) (Pi.mulSingle i (HDiv.hDi …
  -/
  simp_rw [← insertNth_div, ← insertNth_one_right, Pi.div_def, div_self', Pi.one_def]
  /-
    🎉 no goals
  -/


@[simp] lemma smul_empty (x : M) (v : Fin 0 → α) : x • v = ![] := empty_eq _


@[simp] lemma smul_cons (x : M) (y : α) (v : Fin n → α) :
                                                    /-
                                                      α : Type u_1
                                                      M : Type u_2
                                                      n : Nat
                                                      inst✝ : SMul M α
                                                      x : M
                                                      y : α
                                                      v : Fin n → α
                                                      ⊢ Eq (HSMul.hSMul x (Matrix.vecCons y v)) (Matrix.vecCons (HSMul.hSMul x y) (H …
                                                    -/
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/
    x • vecCons y v = vecCons (x • y) (x • v) := by ext i; refine i.cases ?_ ?_ <;> simp
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/


@[simp] lemma empty_add_empty (v w : Fin 0 → α) : v + w = ![] := empty_eq _


@[simp] lemma cons_add (x : α) (v : Fin n → α) (w : Fin n.succ → α) :
    vecCons x v + w = vecCons (x + vecHead w) (v + vecTail w) := by
  /-
    α : Type u_1
    n : Nat
    inst✝ : Add α
    x : α
    v : Fin n → α
    w : Fin n.succ → α
    ⊢ Eq (HAdd.hAdd (Matrix.vecCons x v) w) (Matrix.vecCons (HAdd.hAdd x (Matrix.v …
  -/
                                  /-
                                    🎉 no goals
                                  -/
  ext i; refine i.cases ?_ ?_ <;> simp [vecHead, vecTail]
                                  /-
                                    🎉 no goals
                                  -/


@[simp] lemma add_cons (v : Fin n.succ → α) (y : α) (w : Fin n → α) :
    v + vecCons y w = vecCons (vecHead v + y) (vecTail v + w) := by
  /-
    α : Type u_1
    n : Nat
    inst✝ : Add α
    v : Fin n.succ → α
    y : α
    w : Fin n → α
    ⊢ Eq (HAdd.hAdd v (Matrix.vecCons y w)) (Matrix.vecCons (HAdd.hAdd (Matrix.vec …
  -/
                                  /-
                                    🎉 no goals
                                  -/
  ext i; refine i.cases ?_ ?_ <;> simp [vecHead, vecTail]
                                  /-
                                    🎉 no goals
                                  -/


lemma cons_add_cons (x : α) (v : Fin n → α) (y : α) (w : Fin n → α) :
                                                              /-
                                                                α : Type u_1
                                                                n : Nat
                                                                inst✝ : Add α
                                                                x : α
                                                                v : Fin n → α
                                                                y : α
                                                                w : Fin n → α
                                                                ⊢ Eq (HAdd.hAdd (Matrix.vecCons x v) (Matrix.vecCons y w)) (Matrix.vecCons (HA …
                                                              -/
    vecCons x v + vecCons y w = vecCons (x + y) (v + w) := by simp
                                                              /-
                                                                🎉 no goals
                                                              -/


@[simp] lemma head_add (a b : Fin n.succ → α) : vecHead (a + b) = vecHead a + vecHead b := rfl


@[simp] lemma tail_add (a b : Fin n.succ → α) : vecTail (a + b) = vecTail a + vecTail b := rfl


@[simp] lemma empty_sub_empty (v w : Fin 0 → α) : v - w = ![] := empty_eq _


@[simp] lemma cons_sub (x : α) (v : Fin n → α) (w : Fin n.succ → α) :
    vecCons x v - w = vecCons (x - vecHead w) (v - vecTail w) := by
  /-
    α : Type u_1
    n : Nat
    inst✝ : Sub α
    x : α
    v : Fin n → α
    w : Fin n.succ → α
    ⊢ Eq (HSub.hSub (Matrix.vecCons x v) w) (Matrix.vecCons (HSub.hSub x (Matrix.v …
  -/
                                  /-
                                    🎉 no goals
                                  -/
  ext i; refine i.cases ?_ ?_ <;> simp [vecHead, vecTail]
                                  /-
                                    🎉 no goals
                                  -/


@[simp] lemma sub_cons (v : Fin n.succ → α) (y : α) (w : Fin n → α) :
    v - vecCons y w = vecCons (vecHead v - y) (vecTail v - w) := by
  /-
    α : Type u_1
    n : Nat
    inst✝ : Sub α
    v : Fin n.succ → α
    y : α
    w : Fin n → α
    ⊢ Eq (HSub.hSub v (Matrix.vecCons y w)) (Matrix.vecCons (HSub.hSub (Matrix.vec …
  -/
                                  /-
                                    🎉 no goals
                                  -/
  ext i; refine i.cases ?_ ?_ <;> simp [vecHead, vecTail]
                                  /-
                                    🎉 no goals
                                  -/


lemma cons_sub_cons (x : α) (v : Fin n → α) (y : α) (w : Fin n → α) :
                                                              /-
                                                                α : Type u_1
                                                                n : Nat
                                                                inst✝ : Sub α
                                                                x : α
                                                                v : Fin n → α
                                                                y : α
                                                                w : Fin n → α
                                                                ⊢ Eq (HSub.hSub (Matrix.vecCons x v) (Matrix.vecCons y w)) (Matrix.vecCons (HS …
                                                              -/
    vecCons x v - vecCons y w = vecCons (x - y) (v - w) := by simp
                                                              /-
                                                                🎉 no goals
                                                              -/


@[simp] lemma head_sub (a b : Fin n.succ → α) : vecHead (a - b) = vecHead a - vecHead b := rfl


@[simp] lemma tail_sub (a b : Fin n.succ → α) : vecTail (a - b) = vecTail a - vecTail b := rfl


@[simp] lemma zero_empty : (0 : Fin 0 → α) = ![] := empty_eq _


@[simp] lemma cons_zero_zero : vecCons (0 : α) (0 : Fin n → α) = 0 := by
  /-
    α : Type u_1
    n : Nat
    inst✝ : Zero α
    ⊢ Eq (Matrix.vecCons 0 0) 0
  -/
  ext i; exact i.cases rfl (by simp)
         /-
           🎉 no goals
         -/


@[simp] lemma head_zero : vecHead (0 : Fin n.succ → α) = 0 := rfl


@[simp] lemma tail_zero : vecTail (0 : Fin n.succ → α) = 0 := rfl


@[simp] lemma cons_eq_zero_iff {v : Fin n → α} {x : α} : vecCons x v = 0 ↔ x = 0 ∧ v = 0 where
                             /-
                               α : Type u_1
                               n : Nat
                               inst✝ : Zero α
                               v : Fin n → α
                               x : α
                               h : Eq (Matrix.vecCons x v) 0
                               ⊢ Eq v 0
                             -/
  mp h := ⟨congr_fun h 0, by convert congr_arg vecTail h⟩
                             /-
                               🎉 no goals
                             -/
                           /-
                             α : Type u_1
                             n : Nat
                             inst✝ : Zero α
                             v : Fin n → α
                             x : α
                             x✝ : And (Eq x 0) (Eq v 0)
                             hx : Eq x 0
                             hv : Eq v 0
                             ⊢ Eq (Matrix.vecCons x v) 0
                           -/
  mpr := fun ⟨hx, hv⟩ ↦ by simp [hx, hv]
                           /-
                             🎉 no goals
                           -/


lemma cons_nonzero_iff {v : Fin n → α} {x : α} : vecCons x v ≠ 0 ↔ x ≠ 0 ∨ v ≠ 0 where
  mp h := not_and_or.mp (h ∘ cons_eq_zero_iff.mpr)
  mpr h := mt cons_eq_zero_iff.mp (not_and_or.mpr h)


@[simp] lemma neg_empty (v : Fin 0 → α) : -v = ![] := empty_eq _


@[simp] lemma neg_cons (x : α) (v : Fin n → α) : -vecCons x v = vecCons (-x) (-v) := by
  /-
    α : Type u_1
    n : Nat
    inst✝ : Neg α
    x : α
    v : Fin n → α
    ⊢ Eq (Neg.neg (Matrix.vecCons x v)) (Matrix.vecCons (Neg.neg x) (Neg.neg v))
  -/
                                  /-
                                    🎉 no goals
                                  -/
  ext i; refine i.cases ?_ ?_ <;> simp
                                  /-
                                    🎉 no goals
                                  -/


@[simp] lemma head_neg (a : Fin n.succ → α) : vecHead (-a) = -vecHead a := rfl


@[simp] lemma tail_neg (a : Fin n.succ → α) : vecTail (-a) = -vecTail a := rfl


