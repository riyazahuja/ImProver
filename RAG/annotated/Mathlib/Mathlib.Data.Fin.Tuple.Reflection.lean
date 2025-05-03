/-- Evaluate `FinVec.seq f v = ![(f 0) (v 0), (f 1) (v 1), ...]` -/
def seq : ∀ {m}, (Fin m → α → β) → (Fin m → α) → Fin m → β
  | 0, _, _ => ![]
  | _ + 1, f, v => Matrix.vecCons (f 0 (v 0)) (seq (Matrix.vecTail f) (Matrix.vecTail v))


@[simp]
theorem seq_eq : ∀ {m} (f : Fin m → α → β) (v : Fin m → α), seq f v = fun i => f i (v i)
  | 0, _, _ => Subsingleton.elim _ _
  | n + 1, f, v =>
    funext fun i => by
      /-
        α : Type u_1
        β : Type u_2
        n : Nat
        f : Fin (HAdd.hAdd n 1) → α → β
        v : Fin (HAdd.hAdd n 1) → α
        i : Fin (HAdd.hAdd n 1)
        ⊢ Eq (FinVec.seq f v i) (f i (v i))
      -/
      simp_rw [seq, seq_eq]
      /-
        α : Type u_1
        β : Type u_2
        n : Nat
        f : Fin (HAdd.hAdd n 1) → α → β
        v : Fin (HAdd.hAdd n 1) → α
        i : Fin (HAdd.hAdd n 1)
        ⊢ Eq (Matrix.vecCons (f 0 (v 0)) (fun i => Matrix.vecTail f i (Matrix.vecTail  …
      -/
      refine i.cases ?_ fun i => ?_
        /-
          case refine_1
          α : Type u_1
          β : Type u_2
          n : Nat
          f : Fin (HAdd.hAdd n 1) → α → β
          v : Fin (HAdd.hAdd n 1) → α
          i : Fin (HAdd.hAdd n 1)
          ⊢ Eq (Matrix.vecCons (f 0 (v 0)) (fun i => Matrix.vecTail f i (Matrix.vecTail  …
        -/
      · rfl
        /-
          🎉 no goals
        -/
        /-
          case refine_2
          α : Type u_1
          β : Type u_2
          n : Nat
          f : Fin (HAdd.hAdd n 1) → α → β
          v : Fin (HAdd.hAdd n 1) → α
          i✝ : Fin (HAdd.hAdd n 1)
          i : Fin n
          ⊢ Eq (Matrix.vecCons (f 0 (v 0)) (fun i => Matrix.vecTail f i (Matrix.vecTail  …
        -/
      · rw [Matrix.cons_val_succ]
        /-
          case refine_2
          α : Type u_1
          β : Type u_2
          n : Nat
          f : Fin (HAdd.hAdd n 1) → α → β
          v : Fin (HAdd.hAdd n 1) → α
          i✝ : Fin (HAdd.hAdd n 1)
          i : Fin n
          ⊢ Eq (Matrix.vecTail f i (Matrix.vecTail v i)) (f i.succ (v i.succ))
        -/
        rfl
        /-
          🎉 no goals
        -/


/-- `FinVec.map f v = ![f (v 0), f (v 1), ...]` -/
def map (f : α → β) {m} : (Fin m → α) → Fin m → β :=
  seq fun _ => f


/-- This can be used to prove
```lean
example {f : α → β} (a₁ a₂ : α) : f ∘ ![a₁, a₂] = ![f a₁, f a₂] :=
  (map_eq _ _).symm
```
-/
@[simp]
theorem map_eq (f : α → β) {m} (v : Fin m → α) : map f v = f ∘ v :=
  seq_eq _ _


/-- Expand `v` to `![v 0, v 1, ...]` -/
def etaExpand {m} (v : Fin m → α) : Fin m → α :=
  map id v


/-- This can be used to prove
```lean
example (a : Fin 2 → α) : a = ![a 0, a 1] :=
  (etaExpand_eq _).symm
```
-/
@[simp]
theorem etaExpand_eq {m} (v : Fin m → α) : etaExpand v = v :=
  map_eq id v


/-- `∀` with better defeq for `∀ x : Fin m → α, P x`. -/
def Forall : ∀ {m} (_ : (Fin m → α) → Prop), Prop
  | 0, P => P ![]
  | _ + 1, P => ∀ x : α, Forall fun v => P (Matrix.vecCons x v)


/-- This can be used to prove
```lean
example (P : (Fin 2 → α) → Prop) : (∀ f, P f) ↔ ∀ a₀ a₁, P ![a₀, a₁] :=
  (forall_iff _).symm
```
-/
@[simp]
theorem forall_iff : ∀ {m} (P : (Fin m → α) → Prop), Forall P ↔ ∀ x, P x
  | 0, P => by
    /-
      α : Type u_1
      P : (Fin 0 → α) → Prop
      ⊢ Iff (FinVec.Forall P) (∀ (x : Fin 0 → α), P x)
    -/
    simp only [Forall, Fin.forall_fin_zero_pi]
    /-
      α : Type u_1
      P : (Fin 0 → α) → Prop
      ⊢ Iff (P Matrix.vecEmpty) (P finZeroElim)
    -/
    rfl
    /-
      🎉 no goals
    -/
                     /-
                       α : Type u_1
                       n : Nat
                       P : (Fin n.succ → α) → Prop
                       ⊢ Iff (FinVec.Forall P) (∀ (x : Fin n.succ → α), P x)
                     -/
  | .succ n, P => by simp only [Forall, forall_iff, Fin.forall_fin_succ_pi, Matrix.vecCons]
                     /-
                       🎉 no goals
                     -/


/-- `∃` with better defeq for `∃ x : Fin m → α, P x`. -/
def Exists : ∀ {m} (_ : (Fin m → α) → Prop), Prop
  | 0, P => P ![]
  | _ + 1, P => ∃ x : α, Exists fun v => P (Matrix.vecCons x v)


/-- This can be used to prove
```lean
example (P : (Fin 2 → α) → Prop) : (∃ f, P f) ↔ ∃ a₀ a₁, P ![a₀, a₁] :=
  (exists_iff _).symm
```
-/
theorem exists_iff : ∀ {m} (P : (Fin m → α) → Prop), Exists P ↔ ∃ x, P x
  | 0, P => by
    /-
      α : Type u_1
      P : (Fin 0 → α) → Prop
      ⊢ Iff (FinVec.Exists P) (_root_.Exists fun x => P x)
    -/
    simp only [Exists, Fin.exists_fin_zero_pi, Matrix.vecEmpty]
    /-
      α : Type u_1
      P : (Fin 0 → α) → Prop
      ⊢ Iff (P Fin.elim0) (P finZeroElim)
    -/
    rfl
    /-
      🎉 no goals
    -/
                     /-
                       α : Type u_1
                       n : Nat
                       P : (Fin n.succ → α) → Prop
                       ⊢ Iff (FinVec.Exists P) (_root_.Exists fun x => P x)
                     -/
  | .succ n, P => by simp only [Exists, exists_iff, Fin.exists_fin_succ_pi, Matrix.vecCons]
                     /-
                       🎉 no goals
                     -/


/-- `Finset.univ.sum` with better defeq for `Fin`. -/
def sum [Add α] [Zero α] : ∀ {m} (_ : Fin m → α), α
  | 0, _ => 0
  | 1, v => v 0
  -- Porting note: inline `∘` since it is no longer reducible
  | _ + 2, v => sum (fun i => v (Fin.castSucc i)) + v (Fin.last _)


/-- This can be used to prove
```lean
example [AddCommMonoid α] (a : Fin 3 → α) : ∑ i, a i = a 0 + a 1 + a 2 :=
  (sum_eq _).symm
```
-/
@[simp]
theorem sum_eq [AddCommMonoid α] : ∀ {m} (a : Fin m → α), sum a = ∑ i, a i
  | 0, _ => rfl
  | 1, a => (Fintype.sum_unique a).symm
                   /-
                     α : Type u_1
                     inst✝ : AddCommMonoid α
                     n : Nat
                     a : Fin (HAdd.hAdd n 2) → α
                     ⊢ Eq (FinVec.sum a) (Finset.univ.sum fun i => a i)
                   -/
  | n + 2, a => by rw [Fin.sum_univ_castSucc, sum, sum_eq]
                   /-
                     🎉 no goals
                   -/


