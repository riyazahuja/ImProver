/-- Let `f : Fin (n + 1) → α` be an `(n + 1)`-tuple `(f₀, …, fₙ)` such that
- `f₀ = ⊥` and `fₙ = ⊤`;
- `fₖ₊₁` weakly covers `fₖ` for all `0 ≤ k < n`;
  this means that `fₖ ≤ fₖ₊₁` and there is no `c` such that `fₖ<c<fₖ₊₁`.
Then the range of `f` is a maximal chain. -/
theorem IsMaxChain.range_fin_of_covBy (h0 : f 0 = ⊥) (hlast : f (.last n) = ⊤)
    (hcovBy : ∀ k : Fin n, f k.castSucc ⩿ f k.succ) :
    IsMaxChain (· ≤ ·) (range f) := by
  /-
    α : Type u_1
    inst✝¹ : PartialOrder α
    inst✝ : BoundedOrder α
    n : Nat
    f : Fin (HAdd.hAdd n 1) → α
    h0 : Eq (f 0) Bot.bot
    hlast : Eq (f (Fin.last n)) Top.top
    hcovBy : ∀ (k : Fin n), WCovBy (f k.castSucc) (f k.succ)
    ⊢ IsMaxChain (fun x1 x2 => LE.le x1 x2) (Set.range f)
  -/
  have hmono : Monotone f := Fin.monotone_iff_le_succ.2 fun k ↦ (hcovBy k).1
  /-
    α : Type u_1
    inst✝¹ : PartialOrder α
    inst✝ : BoundedOrder α
    n : Nat
    f : Fin (HAdd.hAdd n 1) → α
    h0 : Eq (f 0) Bot.bot
    hlast : Eq (f (Fin.last n)) Top.top
    hcovBy : ∀ (k : Fin n), WCovBy (f k.castSucc) (f k.succ)
    hmono : Monotone f
    ⊢ IsMaxChain (fun x1 x2 => LE.le x1 x2) (Set.range f)
  -/
  refine ⟨hmono.isChain_range, fun t htc hbt ↦ hbt.antisymm fun x hx ↦ ?_⟩
  /-
    α : Type u_1
    inst✝¹ : PartialOrder α
    inst✝ : BoundedOrder α
    n : Nat
    f : Fin (HAdd.hAdd n 1) → α
    h0 : Eq (f 0) Bot.bot
    hlast : Eq (f (Fin.last n)) Top.top
    hcovBy : ∀ (k : Fin n), WCovBy (f k.castSucc) (f k.succ)
    hmono : Monotone f
    t : Set α
    htc : IsChain (fun x1 x2 => LE.le x1 x2) t
    hbt : HasSubset.Subset (Set.range f) t
    x : α
    hx : Membership.mem t x
    ⊢ Membership.mem (Set.range f) x
  -/
  rw [mem_range]; by_contra! h
  /-
    α : Type u_1
    inst✝¹ : PartialOrder α
    inst✝ : BoundedOrder α
    n : Nat
    f : Fin (HAdd.hAdd n 1) → α
    h0 : Eq (f 0) Bot.bot
    hlast : Eq (f (Fin.last n)) Top.top
    hcovBy : ∀ (k : Fin n), WCovBy (f k.castSucc) (f k.succ)
    hmono : Monotone f
    t : Set α
    htc : IsChain (fun x1 x2 => LE.le x1 x2) t
    hbt : HasSubset.Subset (Set.range f) t
    x : α
    hx : Membership.mem t x
    h : ∀ (y : Fin (HAdd.hAdd n 1)), Ne (f y) x
    ⊢ False
  -/
  suffices ∀ k, f k < x by simpa [hlast] using this (.last _)
  /-
    α : Type u_1
    inst✝¹ : PartialOrder α
    inst✝ : BoundedOrder α
    n : Nat
    f : Fin (HAdd.hAdd n 1) → α
    h0 : Eq (f 0) Bot.bot
    hlast : Eq (f (Fin.last n)) Top.top
    hcovBy : ∀ (k : Fin n), WCovBy (f k.castSucc) (f k.succ)
    hmono : Monotone f
    t : Set α
    htc : IsChain (fun x1 x2 => LE.le x1 x2) t
    hbt : HasSubset.Subset (Set.range f) t
    x : α
    hx : Membership.mem t x
    h : ∀ (y : Fin (HAdd.hAdd n 1)), Ne (f y) x
    ⊢ ∀ (k : Fin (HAdd.hAdd n 1)), LT.lt (f k) x
  -/
  intro k
  induction k using Fin.induction with
  | zero => simpa [h0, bot_lt_iff_ne_bot] using (h 0).symm
  | succ k ihk =>
    rw [range_subset_iff] at hbt
    exact (htc.lt_of_le (hbt k.succ) hx (h _)).resolve_right ((hcovBy k).2 ihk)


/-- Let `f : Fin (n + 1) → α` be an `(n + 1)`-tuple `(f₀, …, fₙ)` such that
- `f₀ = ⊥` and `fₙ = ⊤`;
- `fₖ₊₁` weakly covers `fₖ` for all `0 ≤ k < n`;
  this means that `fₖ ≤ fₖ₊₁` and there is no `c` such that `fₖ<c<fₖ₊₁`.
Then the range of `f` is a `Flag α`. -/
@[simps]
def Flag.rangeFin (f : Fin (n + 1) → α) (h0 : f 0 = ⊥) (hlast : f (.last n) = ⊤)
    (hcovBy : ∀ k : Fin n, f k.castSucc ⩿ f k.succ) : Flag α where
  carrier := range f
  Chain' := (IsMaxChain.range_fin_of_covBy h0 hlast hcovBy).1
  max_chain' := (IsMaxChain.range_fin_of_covBy h0 hlast hcovBy).2


@[simp] theorem Flag.mem_rangeFin {x h0 hlast hcovBy} :
    x ∈ rangeFin f h0 hlast hcovBy ↔ ∃ k, f k = x :=
  Iff.rfl

