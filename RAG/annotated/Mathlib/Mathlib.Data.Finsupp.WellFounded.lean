/-- Transferred from `DFinsupp.Lex.acc`. See the top of that file for an explanation for the
  appearance of the relation `rᶜ ⊓ (≠)`. -/
theorem Lex.acc (hbot : ∀ ⦃n⦄, ¬s n 0) (hs : WellFounded s) (x : α →₀ N)
    (h : ∀ a ∈ x.support, Acc (rᶜ ⊓ (· ≠ ·)) a) :
    Acc (Finsupp.Lex r s) x := by
  /-
    α : Type u_1
    N : Type u_2
    inst✝ : Zero N
    r : α → α → Prop
    s : N → N → Prop
    hbot : ∀ ⦃n : N⦄, Not (s n 0)
    hs : WellFounded s
    x : Finsupp α N
    h : ∀ (a : α), Membership.mem x.support a → Acc (Min.min (HasCompl.compl r) fu …
    ⊢ Acc (Finsupp.Lex r s) x
  -/
  rw [lex_eq_invImage_dfinsupp_lex]
  classical
    refine InvImage.accessible toDFinsupp (DFinsupp.Lex.acc (fun _ => hbot) (fun _ => hs) _ ?_)
    simpa only [toDFinsupp_support] using h


theorem Lex.wellFounded (hbot : ∀ ⦃n⦄, ¬s n 0) (hs : WellFounded s)
    (hr : WellFounded <| rᶜ ⊓ (· ≠ ·)) : WellFounded (Finsupp.Lex r s) :=
  ⟨fun x => Lex.acc hbot hs x fun a _ => hr.apply a⟩


theorem Lex.wellFounded' (hbot : ∀ ⦃n⦄, ¬s n 0) (hs : WellFounded s)
    [IsTrichotomous α r] (hr : WellFounded (Function.swap r)) : WellFounded (Finsupp.Lex r s) :=
  (lex_eq_invImage_dfinsupp_lex r s).symm ▸
    InvImage.wf _ (DFinsupp.Lex.wellFounded' (fun _ => hbot) (fun _ => hs) hr)


instance Lex.wellFoundedLT {α N} [LT α] [IsTrichotomous α (· < ·)] [hα : WellFoundedGT α]
    [CanonicallyOrderedAddCommMonoid N] [hN : WellFoundedLT N] : WellFoundedLT (Lex (α →₀ N)) :=
  ⟨Lex.wellFounded' (fun n => (zero_le n).not_lt) hN.wf hα.wf⟩


theorem Lex.wellFounded_of_finite [IsStrictTotalOrder α r] [Finite α]
    (hs : WellFounded s) : WellFounded (Finsupp.Lex r s) :=
  InvImage.wf (@equivFunOnFinite α N _ _) (Pi.Lex.wellFounded r fun _ => hs)


theorem Lex.wellFoundedLT_of_finite [LinearOrder α] [Finite α] [LT N]
    [hwf : WellFoundedLT N] : WellFoundedLT (Lex (α →₀ N)) :=
  ⟨Finsupp.Lex.wellFounded_of_finite (· < ·) hwf.1⟩


protected theorem wellFoundedLT [Preorder N] [WellFoundedLT N] (hbot : ∀ n : N, ¬n < 0) :
    WellFoundedLT (α →₀ N) :=
  ⟨InvImage.wf toDFinsupp (DFinsupp.wellFoundedLT fun _ a => hbot a).wf⟩


instance wellFoundedLT' {N} [CanonicallyOrderedAddCommMonoid N] [WellFoundedLT N] :
    WellFoundedLT (α →₀ N) :=
  Finsupp.wellFoundedLT fun a => (zero_le a).not_lt


instance wellFoundedLT_of_finite [Finite α] [Preorder N] [WellFoundedLT N] :
    WellFoundedLT (α →₀ N) :=
  ⟨InvImage.wf equivFunOnFinite Function.wellFoundedLT.wf⟩


