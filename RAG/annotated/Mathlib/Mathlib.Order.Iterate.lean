theorem seq_le_seq (hf : Monotone f) (n : ℕ) (h₀ : x 0 ≤ y 0) (hx : ∀ k < n, x (k + 1) ≤ f (x k))
    (hy : ∀ k < n, f (y k) ≤ y (k + 1)) : x n ≤ y n := by
  induction n with
  | zero => exact h₀
  | succ n ihn =>
    refine (hx _ n.lt_succ_self).trans ((hf <| ihn ?_ ?_).trans (hy _ n.lt_succ_self))
    · exact fun k hk => hx _ (hk.trans n.lt_succ_self)
    · exact fun k hk => hy _ (hk.trans n.lt_succ_self)


theorem seq_pos_lt_seq_of_lt_of_le (hf : Monotone f) {n : ℕ} (hn : 0 < n) (h₀ : x 0 ≤ y 0)
    (hx : ∀ k < n, x (k + 1) < f (x k)) (hy : ∀ k < n, f (y k) ≤ y (k + 1)) : x n < y n := by
  induction n with
  | zero => exact hn.false.elim
  | succ n ihn =>
  suffices x n ≤ y n from (hx n n.lt_succ_self).trans_le ((hf this).trans <| hy n n.lt_succ_self)
  cases n with
  | zero => exact h₀
  | succ n =>
    refine (ihn n.zero_lt_succ (fun k hk => hx _ ?_) fun k hk => hy _ ?_).le <;>
    exact hk.trans n.succ.lt_succ_self


theorem seq_pos_lt_seq_of_le_of_lt (hf : Monotone f) {n : ℕ} (hn : 0 < n) (h₀ : x 0 ≤ y 0)
    (hx : ∀ k < n, x (k + 1) ≤ f (x k)) (hy : ∀ k < n, f (y k) < y (k + 1)) : x n < y n :=
  hf.dual.seq_pos_lt_seq_of_lt_of_le hn h₀ hy hx


theorem seq_lt_seq_of_lt_of_le (hf : Monotone f) (n : ℕ) (h₀ : x 0 < y 0)
    (hx : ∀ k < n, x (k + 1) < f (x k)) (hy : ∀ k < n, f (y k) ≤ y (k + 1)) : x n < y n := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    f : α → α
    x y : Nat → α
    hf : Monotone f
    n : Nat
    h₀ : LT.lt (x 0) (y 0)
    hx : ∀ (k : Nat), LT.lt k n → LT.lt (x (HAdd.hAdd k 1)) (f (x k))
    hy : ∀ (k : Nat), LT.lt k n → LE.le (f (y k)) (y (HAdd.hAdd k 1))
    ⊢ LT.lt (x n) (y n)
  -/
  cases n
  /-
    case zero
    α : Type u_1
    inst✝ : Preorder α
    f : α → α
    x y : Nat → α
    hf : Monotone f
    h₀ : LT.lt (x 0) (y 0)
    hx : ∀ (k : Nat), LT.lt k 0 → LT.lt (x (HAdd.hAdd k 1)) (f (x k))
    hy : ∀ (k : Nat), LT.lt k 0 → LE.le (f (y k)) (y (HAdd.hAdd k 1))
    ⊢ LT.lt (x 0) (y 0)
  -/
  exacts [h₀, hf.seq_pos_lt_seq_of_lt_of_le (Nat.zero_lt_succ _) h₀.le hx hy]
  /-
    🎉 no goals
  -/


theorem seq_lt_seq_of_le_of_lt (hf : Monotone f) (n : ℕ) (h₀ : x 0 < y 0)
    (hx : ∀ k < n, x (k + 1) ≤ f (x k)) (hy : ∀ k < n, f (y k) < y (k + 1)) : x n < y n :=
  hf.dual.seq_lt_seq_of_lt_of_le n h₀ hy hx


theorem le_iterate_comp_of_le (hf : Monotone f) (H : h ∘ g ≤ f ∘ h) (n : ℕ) :
    h ∘ g^[n] ≤ f^[n] ∘ h := fun x => by
  /-
    α : Type u_1
    inst✝ : Preorder α
    f : α → α
    β : Type u_2
    g : β → β
    h : β → α
    hf : Monotone f
    H : LE.le (Function.comp h g) (Function.comp f h)
    n : Nat
    x : β
    ⊢ LE.le (Function.comp h (Nat.iterate g n) x) (Function.comp (Nat.iterate f n) …
  -/
  apply hf.seq_le_seq n <;> intros <;>
    /-
      case h₀
      α : Type u_1
      inst✝ : Preorder α
      f : α → α
      β : Type u_2
      g : β → β
      h : β → α
      hf : Monotone f
      H : LE.le (Function.comp h g) (Function.comp f h)
      n : Nat
      x : β
      ⊢ LE.le (Function.comp h (Nat.iterate g 0) x) (Function.comp (Nat.iterate f 0) …
    -/
    /-
      🎉 no goals
    -/
    simp [iterate_succ', -iterate_succ, comp_apply, id_eq, le_refl]
    /-
      🎉 no goals
    -/
  /-
    case hx
    α : Type u_1
    inst✝ : Preorder α
    f : α → α
    β : Type u_2
    g : β → β
    h : β → α
    hf : Monotone f
    H : LE.le (Function.comp h g) (Function.comp f h)
    n : Nat
    x : β
    k✝ : Nat
    a✝ : LT.lt k✝ n
    ⊢ LE.le (h (g (Nat.iterate g k✝ x))) (f (h (Nat.iterate g k✝ x)))
  -/
  case hx => exact H _
  /-
    🎉 no goals
  -/


theorem iterate_comp_le_of_le (hf : Monotone f) (H : f ∘ h ≤ h ∘ g) (n : ℕ) :
    f^[n] ∘ h ≤ h ∘ g^[n] :=
  hf.dual.le_iterate_comp_of_le H n


/-- If `f ≤ g` and `f` is monotone, then `f^[n] ≤ g^[n]`. -/
theorem iterate_le_of_le {g : α → α} (hf : Monotone f) (h : f ≤ g) (n : ℕ) : f^[n] ≤ g^[n] :=
  hf.iterate_comp_le_of_le h n


/-- If `f ≤ g` and `g` is monotone, then `f^[n] ≤ g^[n]`. -/
theorem le_iterate_of_le {g : α → α} (hg : Monotone g) (h : f ≤ g) (n : ℕ) : f^[n] ≤ g^[n] :=
  hg.dual.iterate_le_of_le h n


/-- If $x ≤ f x$ for all $x$ (we write this as `id ≤ f`), then the same is true for any iterate
`f^[n]` of `f`. -/
theorem id_le_iterate_of_id_le (h : id ≤ f) (n : ℕ) : id ≤ f^[n] := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    f : α → α
    h : LE.le id f
    n : Nat
    ⊢ LE.le id (Nat.iterate f n)
  -/
  simpa only [iterate_id] using monotone_id.iterate_le_of_le h n
  /-
    🎉 no goals
  -/


theorem iterate_le_id_of_le_id (h : f ≤ id) (n : ℕ) : f^[n] ≤ id :=
  @id_le_iterate_of_id_le αᵒᵈ _ f h n


theorem monotone_iterate_of_id_le (h : id ≤ f) : Monotone fun m => f^[m] :=
  monotone_nat_of_le_succ fun n x => by
    /-
      α : Type u_1
      inst✝ : Preorder α
      f : α → α
      h : LE.le id f
      n : Nat
      x : α
      ⊢ LE.le (Nat.iterate f n x) (Nat.iterate f (HAdd.hAdd n 1) x)
    -/
    rw [iterate_succ_apply']
    /-
      α : Type u_1
      inst✝ : Preorder α
      f : α → α
      h : LE.le id f
      n : Nat
      x : α
      ⊢ LE.le (Nat.iterate f n x) (f (Nat.iterate f n x))
    -/
    exact h _
    /-
      🎉 no goals
    -/


theorem antitone_iterate_of_le_id (h : f ≤ id) : Antitone fun m => f^[m] := fun m n hmn =>
  @monotone_iterate_of_id_le αᵒᵈ _ f h m n hmn


theorem iterate_le_of_map_le (h : Commute f g) (hf : Monotone f) (hg : Monotone g) {x}
    (hx : f x ≤ g x) (n : ℕ) : f^[n] x ≤ g^[n] x := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    f g : α → α
    h : Function.Commute f g
    hf : Monotone f
    hg : Monotone g
    x : α
    hx : LE.le (f x) (g x)
    n : Nat
    ⊢ LE.le (Nat.iterate f n x) (Nat.iterate g n x)
  -/
  apply hf.seq_le_seq n
    /-
      case h₀
      α : Type u_1
      inst✝ : Preorder α
      f g : α → α
      h : Function.Commute f g
      hf : Monotone f
      hg : Monotone g
      x : α
      hx : LE.le (f x) (g x)
      n : Nat
      ⊢ LE.le (Nat.iterate f 0 x) (Nat.iterate g 0 x)
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case hx
      α : Type u_1
      inst✝ : Preorder α
      f g : α → α
      h : Function.Commute f g
      hf : Monotone f
      hg : Monotone g
      x : α
      hx : LE.le (f x) (g x)
      n : Nat
      ⊢ ∀ (k : Nat), LT.lt k n → LE.le (Nat.iterate f (HAdd.hAdd k 1) x) (f (Nat.ite …
    -/
  · intros; rw [iterate_succ_apply']
            /-
              🎉 no goals
            -/
    /-
      case hy
      α : Type u_1
      inst✝ : Preorder α
      f g : α → α
      h : Function.Commute f g
      hf : Monotone f
      hg : Monotone g
      x : α
      hx : LE.le (f x) (g x)
      n : Nat
      ⊢ ∀ (k : Nat), LT.lt k n → LE.le (f (Nat.iterate g k x)) (Nat.iterate g (HAdd. …
    -/
  · intros; simp [h.iterate_right _ _, hg.iterate _ hx]
            /-
              🎉 no goals
            -/


theorem iterate_pos_lt_of_map_lt (h : Commute f g) (hf : Monotone f) (hg : StrictMono g) {x}
    (hx : f x < g x) {n} (hn : 0 < n) : f^[n] x < g^[n] x := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    f g : α → α
    h : Function.Commute f g
    hf : Monotone f
    hg : StrictMono g
    x : α
    hx : LT.lt (f x) (g x)
    n : Nat
    hn : LT.lt 0 n
    ⊢ LT.lt (Nat.iterate f n x) (Nat.iterate g n x)
  -/
  apply hf.seq_pos_lt_seq_of_le_of_lt hn
    /-
      case h₀
      α : Type u_1
      inst✝ : Preorder α
      f g : α → α
      h : Function.Commute f g
      hf : Monotone f
      hg : StrictMono g
      x : α
      hx : LT.lt (f x) (g x)
      n : Nat
      hn : LT.lt 0 n
      ⊢ LE.le (Nat.iterate f 0 x) (Nat.iterate g 0 x)
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case hx
      α : Type u_1
      inst✝ : Preorder α
      f g : α → α
      h : Function.Commute f g
      hf : Monotone f
      hg : StrictMono g
      x : α
      hx : LT.lt (f x) (g x)
      n : Nat
      hn : LT.lt 0 n
      ⊢ ∀ (k : Nat), LT.lt k n → LE.le (Nat.iterate f (HAdd.hAdd k 1) x) (f (Nat.ite …
    -/
  · intros; rw [iterate_succ_apply']
            /-
              🎉 no goals
            -/
    /-
      case hy
      α : Type u_1
      inst✝ : Preorder α
      f g : α → α
      h : Function.Commute f g
      hf : Monotone f
      hg : StrictMono g
      x : α
      hx : LT.lt (f x) (g x)
      n : Nat
      hn : LT.lt 0 n
      ⊢ ∀ (k : Nat), LT.lt k n → LT.lt (f (Nat.iterate g k x)) (Nat.iterate g (HAdd. …
    -/
  · intros; simp [h.iterate_right _ _, hg.iterate _ hx]
            /-
              🎉 no goals
            -/


theorem iterate_pos_lt_of_map_lt' (h : Commute f g) (hf : StrictMono f) (hg : Monotone g) {x}
    (hx : f x < g x) {n} (hn : 0 < n) : f^[n] x < g^[n] x :=
  @iterate_pos_lt_of_map_lt αᵒᵈ _ g f h.symm hg.dual hf.dual x hx n hn


theorem iterate_pos_lt_iff_map_lt (h : Commute f g) (hf : Monotone f) (hg : StrictMono g) {x n}
    (hn : 0 < n) : f^[n] x < g^[n] x ↔ f x < g x := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    f g : α → α
    h : Function.Commute f g
    hf : Monotone f
    hg : StrictMono g
    x : α
    n : Nat
    hn : LT.lt 0 n
    ⊢ Iff (LT.lt (Nat.iterate f n x) (Nat.iterate g n x)) (LT.lt (f x) (g x))
  -/
  rcases lt_trichotomy (f x) (g x) with (H | H | H)
    /-
      case inl
      α : Type u_1
      inst✝ : LinearOrder α
      f g : α → α
      h : Function.Commute f g
      hf : Monotone f
      hg : StrictMono g
      x : α
      n : Nat
      hn : LT.lt 0 n
      H : LT.lt (f x) (g x)
      ⊢ Iff (LT.lt (Nat.iterate f n x) (Nat.iterate g n x)) (LT.lt (f x) (g x))
    -/
  · simp only [*, iterate_pos_lt_of_map_lt]
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      α : Type u_1
      inst✝ : LinearOrder α
      f g : α → α
      h : Function.Commute f g
      hf : Monotone f
      hg : StrictMono g
      x : α
      n : Nat
      hn : LT.lt 0 n
      H : Eq (f x) (g x)
      ⊢ Iff (LT.lt (Nat.iterate f n x) (Nat.iterate g n x)) (LT.lt (f x) (g x))
    -/
  · simp only [*, h.iterate_eq_of_map_eq, lt_irrefl]
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      α : Type u_1
      inst✝ : LinearOrder α
      f g : α → α
      h : Function.Commute f g
      hf : Monotone f
      hg : StrictMono g
      x : α
      n : Nat
      hn : LT.lt 0 n
      H : LT.lt (g x) (f x)
      ⊢ Iff (LT.lt (Nat.iterate f n x) (Nat.iterate g n x)) (LT.lt (f x) (g x))
    -/
  · simp only [lt_asymm H, lt_asymm (h.symm.iterate_pos_lt_of_map_lt' hg hf H hn)]
    /-
      🎉 no goals
    -/


theorem iterate_pos_lt_iff_map_lt' (h : Commute f g) (hf : StrictMono f) (hg : Monotone g) {x n}
    (hn : 0 < n) : f^[n] x < g^[n] x ↔ f x < g x :=
  @iterate_pos_lt_iff_map_lt αᵒᵈ _ _ _ h.symm hg.dual hf.dual x n hn


theorem iterate_pos_le_iff_map_le (h : Commute f g) (hf : Monotone f) (hg : StrictMono g) {x n}
    (hn : 0 < n) : f^[n] x ≤ g^[n] x ↔ f x ≤ g x := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    f g : α → α
    h : Function.Commute f g
    hf : Monotone f
    hg : StrictMono g
    x : α
    n : Nat
    hn : LT.lt 0 n
    ⊢ Iff (LE.le (Nat.iterate f n x) (Nat.iterate g n x)) (LE.le (f x) (g x))
  -/
  simpa only [not_lt] using not_congr (h.symm.iterate_pos_lt_iff_map_lt' hg hf hn)
  /-
    🎉 no goals
  -/


theorem iterate_pos_le_iff_map_le' (h : Commute f g) (hf : StrictMono f) (hg : Monotone g) {x n}
    (hn : 0 < n) : f^[n] x ≤ g^[n] x ↔ f x ≤ g x := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    f g : α → α
    h : Function.Commute f g
    hf : StrictMono f
    hg : Monotone g
    x : α
    n : Nat
    hn : LT.lt 0 n
    ⊢ Iff (LE.le (Nat.iterate f n x) (Nat.iterate g n x)) (LE.le (f x) (g x))
  -/
  simpa only [not_lt] using not_congr (h.symm.iterate_pos_lt_iff_map_lt hg hf hn)
  /-
    🎉 no goals
  -/


theorem iterate_pos_eq_iff_map_eq (h : Commute f g) (hf : Monotone f) (hg : StrictMono g) {x n}
    (hn : 0 < n) : f^[n] x = g^[n] x ↔ f x = g x := by
  simp only [le_antisymm_iff, h.iterate_pos_le_iff_map_le hf hg hn,
    h.symm.iterate_pos_le_iff_map_le' hg hf hn]


/-- If `f` is a monotone map and `x ≤ f x` at some point `x`, then the iterates `f^[n] x` form
a monotone sequence. -/
theorem monotone_iterate_of_le_map (hf : Monotone f) (hx : x ≤ f x) : Monotone fun n => f^[n] x :=
  monotone_nat_of_le_succ fun n => by
    /-
      α : Type u_1
      inst✝ : Preorder α
      f : α → α
      x : α
      hf : Monotone f
      hx : LE.le x (f x)
      n : Nat
      ⊢ LE.le (Nat.iterate f n x) (Nat.iterate f (HAdd.hAdd n 1) x)
    -/
    rw [iterate_succ_apply]
    /-
      α : Type u_1
      inst✝ : Preorder α
      f : α → α
      x : α
      hf : Monotone f
      hx : LE.le x (f x)
      n : Nat
      ⊢ LE.le (Nat.iterate f n x) (Nat.iterate f n (f x))
    -/
    exact hf.iterate n hx
    /-
      🎉 no goals
    -/


/-- If `f` is a monotone map and `f x ≤ x` at some point `x`, then the iterates `f^[n] x` form
an antitone sequence. -/
theorem antitone_iterate_of_map_le (hf : Monotone f) (hx : f x ≤ x) : Antitone fun n => f^[n] x :=
  hf.dual.monotone_iterate_of_le_map hx


/-- If `f` is a strictly monotone map and `x < f x` at some point `x`, then the iterates `f^[n] x`
form a strictly monotone sequence. -/
theorem strictMono_iterate_of_lt_map (hf : StrictMono f) (hx : x < f x) :
    StrictMono fun n => f^[n] x :=
  strictMono_nat_of_lt_succ fun n => by
    /-
      α : Type u_1
      inst✝ : Preorder α
      f : α → α
      x : α
      hf : StrictMono f
      hx : LT.lt x (f x)
      n : Nat
      ⊢ LT.lt (Nat.iterate f n x) (Nat.iterate f (HAdd.hAdd n 1) x)
    -/
    rw [iterate_succ_apply]
    /-
      α : Type u_1
      inst✝ : Preorder α
      f : α → α
      x : α
      hf : StrictMono f
      hx : LT.lt x (f x)
      n : Nat
      ⊢ LT.lt (Nat.iterate f n x) (Nat.iterate f n (f x))
    -/
    exact hf.iterate n hx
    /-
      🎉 no goals
    -/


/-- If `f` is a strictly antitone map and `f x < x` at some point `x`, then the iterates `f^[n] x`
form a strictly antitone sequence. -/
theorem strictAnti_iterate_of_map_lt (hf : StrictMono f) (hx : f x < x) :
    StrictAnti fun n => f^[n] x :=
  hf.dual.strictMono_iterate_of_lt_map hx


