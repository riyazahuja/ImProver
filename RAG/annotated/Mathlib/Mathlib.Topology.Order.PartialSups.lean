protected lemma partialSups (hf : ∀ k ≤ n, Tendsto (f k) l (𝓝 (g k))) :
    Tendsto (partialSups f n) l (𝓝 (partialSups g n)) := by
  /-
    L : Type u_1
    inst✝² : SemilatticeSup L
    inst✝¹ : TopologicalSpace L
    inst✝ : ContinuousSup L
    α : Type u_2
    l : Filter α
    f : Nat → α → L
    g : Nat → L
    n : Nat
    hf : ∀ (k : Nat), LE.le k n → Filter.Tendsto (f k) l (nhds (g k))
    ⊢ Filter.Tendsto ((partialSups f) n) l (nhds ((partialSups g) n))
  -/
  simp only [partialSups_eq_sup'_range]
  /-
    L : Type u_1
    inst✝² : SemilatticeSup L
    inst✝¹ : TopologicalSpace L
    inst✝ : ContinuousSup L
    α : Type u_2
    l : Filter α
    f : Nat → α → L
    g : Nat → L
    n : Nat
    hf : ∀ (k : Nat), LE.le k n → Filter.Tendsto (f k) l (nhds (g k))
    ⊢ Filter.Tendsto ((Finset.range (HAdd.hAdd n 1)).sup' ⋯ f) l (nhds ((Finset.ra …
  -/
  refine finset_sup'_nhds _ ?_
  /-
    L : Type u_1
    inst✝² : SemilatticeSup L
    inst✝¹ : TopologicalSpace L
    inst✝ : ContinuousSup L
    α : Type u_2
    l : Filter α
    f : Nat → α → L
    g : Nat → L
    n : Nat
    hf : ∀ (k : Nat), LE.le k n → Filter.Tendsto (f k) l (nhds (g k))
    ⊢ ∀ (i : Nat), Membership.mem (Finset.range (HAdd.hAdd n 1)) i → Filter.Tendst …
  -/
  simpa [Nat.lt_succ_iff]
  /-
    🎉 no goals
  -/


protected lemma partialSups_apply (hf : ∀ k ≤ n, Tendsto (f k) l (𝓝 (g k))) :
    Tendsto (fun a ↦ partialSups (f · a) n) l (𝓝 (partialSups g n)) := by
  /-
    L : Type u_1
    inst✝² : SemilatticeSup L
    inst✝¹ : TopologicalSpace L
    inst✝ : ContinuousSup L
    α : Type u_2
    l : Filter α
    f : Nat → α → L
    g : Nat → L
    n : Nat
    hf : ∀ (k : Nat), LE.le k n → Filter.Tendsto (f k) l (nhds (g k))
    ⊢ Filter.Tendsto (fun a => (partialSups fun x => f x a) n) l (nhds ((partialSu …
  -/
  simpa only [← partialSups_apply] using Tendsto.partialSups hf
  /-
    🎉 no goals
  -/


protected lemma ContinuousAt.partialSups_apply (hf : ∀ k ≤ n, ContinuousAt (f k) x) :
    ContinuousAt (fun a ↦ partialSups (f · a) n) x :=
  Tendsto.partialSups_apply hf


protected lemma ContinuousAt.partialSups (hf : ∀ k ≤ n, ContinuousAt (f k) x) :
    ContinuousAt (partialSups f n) x := by
  /-
    L : Type u_1
    inst✝³ : SemilatticeSup L
    inst✝² : TopologicalSpace L
    inst✝¹ : ContinuousSup L
    X : Type u_2
    inst✝ : TopologicalSpace X
    f : Nat → X → L
    n : Nat
    x : X
    hf : ∀ (k : Nat), LE.le k n → ContinuousAt (f k) x
    ⊢ ContinuousAt ((partialSups f) n) x
  -/
  simpa only [← partialSups_apply] using ContinuousAt.partialSups_apply hf
  /-
    🎉 no goals
  -/


protected lemma ContinuousWithinAt.partialSups_apply (hf : ∀ k ≤ n, ContinuousWithinAt (f k) s x) :
    ContinuousWithinAt (fun a ↦ partialSups (f · a) n) s x :=
  Tendsto.partialSups_apply hf


protected lemma ContinuousWithinAt.partialSups (hf : ∀ k ≤ n, ContinuousWithinAt (f k) s x) :
    ContinuousWithinAt (partialSups f n) s x := by
  /-
    L : Type u_1
    inst✝³ : SemilatticeSup L
    inst✝² : TopologicalSpace L
    inst✝¹ : ContinuousSup L
    X : Type u_2
    inst✝ : TopologicalSpace X
    f : Nat → X → L
    n : Nat
    s : Set X
    x : X
    hf : ∀ (k : Nat), LE.le k n → ContinuousWithinAt (f k) s x
    ⊢ ContinuousWithinAt ((partialSups f) n) s x
  -/
  simpa only [← partialSups_apply] using ContinuousWithinAt.partialSups_apply hf
  /-
    🎉 no goals
  -/


protected lemma ContinuousOn.partialSups_apply (hf : ∀ k ≤ n, ContinuousOn (f k) s) :
    ContinuousOn (fun a ↦ partialSups (f · a) n) s := fun x hx ↦
  ContinuousWithinAt.partialSups_apply fun k hk ↦ hf k hk x hx


protected lemma ContinuousOn.partialSups (hf : ∀ k ≤ n, ContinuousOn (f k) s) :
    ContinuousOn (partialSups f n) s := fun x hx ↦
  ContinuousWithinAt.partialSups fun k hk ↦ hf k hk x hx


protected lemma Continuous.partialSups_apply (hf : ∀ k ≤ n, Continuous (f k)) :
    Continuous (fun a ↦ partialSups (f · a) n) :=
  continuous_iff_continuousAt.2 fun _ ↦ ContinuousAt.partialSups_apply fun k hk ↦
    (hf k hk).continuousAt


protected lemma Continuous.partialSups (hf : ∀ k ≤ n, Continuous (f k)) :
    Continuous (partialSups f n) :=
  continuous_iff_continuousAt.2 fun _ ↦ ContinuousAt.partialSups fun k hk ↦ (hf k hk).continuousAt

