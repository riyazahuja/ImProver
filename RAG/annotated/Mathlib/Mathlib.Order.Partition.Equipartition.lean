/-- An equipartition is a partition whose parts are all the same size, up to a difference of `1`. -/
def IsEquipartition : Prop :=
  (P.parts : Set (Finset α)).EquitableOn card


theorem isEquipartition_iff_card_parts_eq_average :
    P.IsEquipartition ↔
      ∀ a : Finset α, a ∈ P.parts → #a = #s / #P.parts ∨ #a = #s / #P.parts + 1 := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    P : Finpartition s
    ⊢ Iff P.IsEquipartition (∀ (a : Finset α), Membership.mem P.parts a → Or (Eq a …
  -/
  simp_rw [IsEquipartition, Finset.equitableOn_iff, P.sum_card_parts]
  /-
    🎉 no goals
  -/


lemma not_isEquipartition :
    ¬P.IsEquipartition ↔ ∃ a ∈ P.parts, ∃ b ∈ P.parts, #b + 1 < #a := Set.not_equitableOn


theorem _root_.Set.Subsingleton.isEquipartition (h : (P.parts : Set (Finset α)).Subsingleton) :
    P.IsEquipartition :=
  Set.Subsingleton.equitableOn h _


theorem IsEquipartition.card_parts_eq_average (hP : P.IsEquipartition) (ht : t ∈ P.parts) :
    #t = #s / #P.parts ∨ #t = #s / #P.parts + 1 :=
  P.isEquipartition_iff_card_parts_eq_average.1 hP _ ht


theorem IsEquipartition.card_part_eq_average_iff (hP : P.IsEquipartition) (ht : t ∈ P.parts) :
    #t = #s / #P.parts ↔ #t ≠ #s / #P.parts + 1 := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Finset α
    P : Finpartition s
    hP : P.IsEquipartition
    ht : Membership.mem P.parts t
    ⊢ Iff (Eq t.card (HDiv.hDiv s.card P.parts.card)) (Ne t.card (HAdd.hAdd (HDiv. …
  -/
  have a := hP.card_parts_eq_average ht
  have b : ¬(#t = #s / #P.parts ∧ #t = #s / #P.parts + 1) := by
    by_contra h; exact absurd (h.1 ▸ h.2) (lt_add_one _).ne
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Finset α
    P : Finpartition s
    hP : P.IsEquipartition
    ht : Membership.mem P.parts t
    a : Or (Eq t.card (HDiv.hDiv s.card P.parts.card)) (Eq t.card (HAdd.hAdd (HDiv …
    b : Not (And (Eq t.card (HDiv.hDiv s.card P.parts.card)) (Eq t.card (HAdd.hAdd …
    ⊢ Iff (Eq t.card (HDiv.hDiv s.card P.parts.card)) (Ne t.card (HAdd.hAdd (HDiv. …
  -/
  tauto
  /-
    🎉 no goals
  -/


theorem IsEquipartition.average_le_card_part (hP : P.IsEquipartition) (ht : t ∈ P.parts) :
    #s / #P.parts ≤ #t := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Finset α
    P : Finpartition s
    hP : P.IsEquipartition
    ht : Membership.mem P.parts t
    ⊢ LE.le (HDiv.hDiv s.card P.parts.card) t.card
  -/
  rw [← P.sum_card_parts]
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Finset α
    P : Finpartition s
    hP : P.IsEquipartition
    ht : Membership.mem P.parts t
    ⊢ LE.le (HDiv.hDiv (P.parts.sum fun i => i.card) P.parts.card) t.card
  -/
  exact Finset.EquitableOn.le hP ht
  /-
    🎉 no goals
  -/


theorem IsEquipartition.card_part_le_average_add_one (hP : P.IsEquipartition) (ht : t ∈ P.parts) :
    #t ≤ #s / #P.parts + 1 := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Finset α
    P : Finpartition s
    hP : P.IsEquipartition
    ht : Membership.mem P.parts t
    ⊢ LE.le t.card (HAdd.hAdd (HDiv.hDiv s.card P.parts.card) 1)
  -/
  rw [← P.sum_card_parts]
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Finset α
    P : Finpartition s
    hP : P.IsEquipartition
    ht : Membership.mem P.parts t
    ⊢ LE.le t.card (HAdd.hAdd (HDiv.hDiv (P.parts.sum fun i => i.card) P.parts.car …
  -/
  exact Finset.EquitableOn.le_add_one hP ht
  /-
    🎉 no goals
  -/


theorem IsEquipartition.filter_ne_average_add_one_eq_average (hP : P.IsEquipartition) :
    {p ∈ P.parts | ¬#p = #s / #P.parts + 1} = {p ∈ P.parts | #p = #s / #P.parts} := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    P : Finpartition s
    hP : P.IsEquipartition
    ⊢ Eq (Finset.filter (fun p => Not (Eq p.card (HAdd.hAdd (HDiv.hDiv s.card P.pa …
  -/
  ext p
  /-
    case h
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    P : Finpartition s
    hP : P.IsEquipartition
    p : Finset α
    ⊢ Iff (Membership.mem (Finset.filter (fun p => Not (Eq p.card (HAdd.hAdd (HDiv …
  -/
  simp only [mem_filter, and_congr_right_iff]
  /-
    case h
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    P : Finpartition s
    hP : P.IsEquipartition
    p : Finset α
    ⊢ Membership.mem P.parts p → Iff (Not (Eq p.card (HAdd.hAdd (HDiv.hDiv s.card  …
  -/
  exact fun hp ↦ (hP.card_part_eq_average_iff hp).symm
  /-
    🎉 no goals
  -/


/-- An equipartition of a finset with `n` elements into `k` parts has
`n % k` parts of size `n / k + 1`. -/
theorem IsEquipartition.card_large_parts_eq_mod (hP : P.IsEquipartition) :
    #{p ∈ P.parts | #p = #s / #P.parts + 1} = #s % #P.parts := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    P : Finpartition s
    hP : P.IsEquipartition
    ⊢ Eq (Finset.filter (fun p => Eq p.card (HAdd.hAdd (HDiv.hDiv s.card P.parts.c …
  -/
  have z := P.sum_card_parts
  rw [← sum_filter_add_sum_filter_not (s := P.parts) (p := fun x ↦ #x = #s / #P.parts + 1),
    hP.filter_ne_average_add_one_eq_average, sum_const_nat (m := #s / #P.parts + 1) (by simp),
    sum_const_nat (m := #s / #P.parts) (by simp), ← hP.filter_ne_average_add_one_eq_average,
    mul_add, add_comm, ← add_assoc, ← add_mul, mul_one, add_comm #_,
    filter_card_add_filter_neg_card_eq_card, add_comm] at z
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    P : Finpartition s
    hP : P.IsEquipartition
    z : Eq (HAdd.hAdd (Finset.filter (fun x => Eq x.card (HAdd.hAdd (HDiv.hDiv s.c …
    ⊢ Eq (Finset.filter (fun p => Eq p.card (HAdd.hAdd (HDiv.hDiv s.card P.parts.c …
  -/
  rw [← add_left_inj, Nat.mod_add_div, z]
  /-
    🎉 no goals
  -/


/-- An equipartition of a finset with `n` elements into `k` parts has
`n - n % k` parts of size `n / k`. -/
theorem IsEquipartition.card_small_parts_eq_mod (hP : P.IsEquipartition) :
    #{p ∈ P.parts | #p = #s / #P.parts} = #P.parts - #s % #P.parts := by
  conv_rhs =>
    arg 1
    rw [← filter_card_add_filter_neg_card_eq_card (p := fun p ↦ #p = #s / #P.parts + 1)]
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    P : Finpartition s
    hP : P.IsEquipartition
    ⊢ Eq (Finset.filter (fun p => Eq p.card (HDiv.hDiv s.card P.parts.card)) P.par …
  -/
  rw [hP.card_large_parts_eq_mod, add_tsub_cancel_left, hP.filter_ne_average_add_one_eq_average]
  /-
    🎉 no goals
  -/


/-- There exists an enumeration of an equipartition's parts where
larger parts map to smaller numbers and vice versa. -/
theorem IsEquipartition.exists_partsEquiv (hP : P.IsEquipartition) :
    ∃ f : P.parts ≃ Fin #P.parts, ∀ t, #t.1 = #s / #P.parts + 1 ↔ f t < #s % #P.parts := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    P : Finpartition s
    hP : P.IsEquipartition
    ⊢ Exists fun f => ∀ (t : Subtype fun x => Membership.mem P.parts x), Iff (Eq ( …
  -/
  let el := {p ∈ P.parts | #p = #s / #P.parts + 1}.equivFin
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    P : Finpartition s
    hP : P.IsEquipartition
    el : Equiv (Subtype fun x => Membership.mem (Finset.filter (fun p => Eq p.card …
    ⊢ Exists fun f => ∀ (t : Subtype fun x => Membership.mem P.parts x), Iff (Eq ( …
  -/
  let es := {p ∈ P.parts | #p = #s / #P.parts}.equivFin
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    P : Finpartition s
    hP : P.IsEquipartition
    el : Equiv (Subtype fun x => Membership.mem (Finset.filter (fun p => Eq p.card …
    es : Equiv (Subtype fun x => Membership.mem (Finset.filter (fun p => Eq p.card …
    ⊢ Exists fun f => ∀ (t : Subtype fun x => Membership.mem P.parts x), Iff (Eq ( …
  -/
  simp_rw [mem_filter, hP.card_large_parts_eq_mod] at el
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    P : Finpartition s
    hP : P.IsEquipartition
    es : Equiv (Subtype fun x => Membership.mem (Finset.filter (fun p => Eq p.card …
    el : Equiv (Subtype fun x => And (Membership.mem P.parts x) (Eq x.card (HAdd.h …
    ⊢ Exists fun f => ∀ (t : Subtype fun x => Membership.mem P.parts x), Iff (Eq ( …
  -/
  simp_rw [mem_filter, hP.card_small_parts_eq_mod] at es
  let sneg :
      {x // x ∈ P.parts ∧ ¬#x = #s / #P.parts + 1} ≃ {x // x ∈ P.parts ∧ #x = #s / #P.parts} := by
    apply (Equiv.refl _).subtypeEquiv
    simp only [Equiv.refl_apply, and_congr_right_iff]
    exact fun _ ha ↦ by rw [hP.card_part_eq_average_iff ha, ne_eq]
  replace el : { x : P.parts // #x.1 = #s / #P.parts + 1 } ≃
      Fin (#s % #P.parts) := (Equiv.Set.sep ..).symm.trans el
  replace es : { x : P.parts // ¬#x.1 = #s / #P.parts + 1 } ≃
      Fin (#P.parts - #s % #P.parts) := (Equiv.Set.sep ..).symm.trans (sneg.trans es)
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    P : Finpartition s
    hP : P.IsEquipartition
    sneg : Equiv (Subtype fun x => And (Membership.mem P.parts x) (Not (Eq x.card  …
    el : Equiv (Subtype fun x => Eq (↑x).card (HAdd.hAdd (HDiv.hDiv s.card P.parts …
    es : Equiv (Subtype fun x => Not (Eq (↑x).card (HAdd.hAdd (HDiv.hDiv s.card P. …
    ⊢ Exists fun f => ∀ (t : Subtype fun x => Membership.mem P.parts x), Iff (Eq ( …
  -/
  let f := (Equiv.sumCompl _).symm.trans ((el.sumCongr es).trans finSumFinEquiv)
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    P : Finpartition s
    hP : P.IsEquipartition
    sneg : Equiv (Subtype fun x => And (Membership.mem P.parts x) (Not (Eq x.card  …
    el : Equiv (Subtype fun x => Eq (↑x).card (HAdd.hAdd (HDiv.hDiv s.card P.parts …
    es : Equiv (Subtype fun x => Not (Eq (↑x).card (HAdd.hAdd (HDiv.hDiv s.card P. …
    f : Equiv (Subtype fun x => Membership.mem P.parts x) (Fin (HAdd.hAdd (HMod.hM …
    ⊢ Exists fun f => ∀ (t : Subtype fun x => Membership.mem P.parts x), Iff (Eq ( …
  -/
  use f.trans (finCongr (Nat.add_sub_of_le P.card_mod_card_parts_le))
  /-
    case h
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    P : Finpartition s
    hP : P.IsEquipartition
    sneg : Equiv (Subtype fun x => And (Membership.mem P.parts x) (Not (Eq x.card  …
    el : Equiv (Subtype fun x => Eq (↑x).card (HAdd.hAdd (HDiv.hDiv s.card P.parts …
    es : Equiv (Subtype fun x => Not (Eq (↑x).card (HAdd.hAdd (HDiv.hDiv s.card P. …
    f : Equiv (Subtype fun x => Membership.mem P.parts x) (Fin (HAdd.hAdd (HMod.hM …
    ⊢ ∀ (t : Subtype fun x => Membership.mem P.parts x), Iff (Eq (↑t).card (HAdd.h …
  -/
  intro ⟨p, _⟩
  /-
    case h
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    P : Finpartition s
    hP : P.IsEquipartition
    sneg : Equiv (Subtype fun x => And (Membership.mem P.parts x) (Not (Eq x.card  …
    el : Equiv (Subtype fun x => Eq (↑x).card (HAdd.hAdd (HDiv.hDiv s.card P.parts …
    es : Equiv (Subtype fun x => Not (Eq (↑x).card (HAdd.hAdd (HDiv.hDiv s.card P. …
    f : Equiv (Subtype fun x => Membership.mem P.parts x) (Fin (HAdd.hAdd (HMod.hM …
    p : Finset α
    property✝ : Membership.mem P.parts p
    ⊢ Iff (Eq (↑⟨p, property✝⟩).card (HAdd.hAdd (HDiv.hDiv s.card P.parts.card) 1) …
  -/
  simp_rw [f, Equiv.trans_apply, Equiv.sumCongr_apply, finCongr_apply, Fin.coe_cast]
  /-
    case h
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    P : Finpartition s
    hP : P.IsEquipartition
    sneg : Equiv (Subtype fun x => And (Membership.mem P.parts x) (Not (Eq x.card  …
    el : Equiv (Subtype fun x => Eq (↑x).card (HAdd.hAdd (HDiv.hDiv s.card P.parts …
    es : Equiv (Subtype fun x => Not (Eq (↑x).card (HAdd.hAdd (HDiv.hDiv s.card P. …
    f : Equiv (Subtype fun x => Membership.mem P.parts x) (Fin (HAdd.hAdd (HMod.hM …
    p : Finset α
    property✝ : Membership.mem P.parts p
    ⊢ Iff (Eq p.card (HAdd.hAdd (HDiv.hDiv s.card P.parts.card) 1)) (LT.lt (↑(finS …
  -/
                                           /-
                                             🎉 no goals
                                           -/
  by_cases hc : #p = #s / #P.parts + 1 <;> simp [hc]
                                           /-
                                             🎉 no goals
                                           -/


/-- Given a finset equipartitioned into `k` parts, its elements can be enumerated such that
elements in the same part have congruent indices modulo `k`. -/
theorem IsEquipartition.exists_partPreservingEquiv (hP : P.IsEquipartition) : ∃ f : s ≃ Fin #s,
    ∀ a b : s, P.part a = P.part b ↔ f a % #P.parts = f b % #P.parts := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    P : Finpartition s
    hP : P.IsEquipartition
    ⊢ Exists fun f => ∀ (a b : Subtype fun x => Membership.mem s x), Iff (Eq (P.pa …
  -/
  obtain ⟨f, hf⟩ := P.exists_enumeration
  /-
    case intro
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    P : Finpartition s
    hP : P.IsEquipartition
    f : Equiv (Subtype fun x => Membership.mem s x) (Sigma fun t => Fin (↑t).card)
    hf : ∀ (a b : Subtype fun x => Membership.mem s x), Iff (Eq (P.part ↑a) (P.par …
    ⊢ Exists fun f => ∀ (a b : Subtype fun x => Membership.mem s x), Iff (Eq (P.pa …
  -/
  obtain ⟨g, hg⟩ := hP.exists_partsEquiv
  /-
    case intro.intro
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    P : Finpartition s
    hP : P.IsEquipartition
    f : Equiv (Subtype fun x => Membership.mem s x) (Sigma fun t => Fin (↑t).card)
    hf : ∀ (a b : Subtype fun x => Membership.mem s x), Iff (Eq (P.part ↑a) (P.par …
    g : Equiv (Subtype fun x => Membership.mem P.parts x) (Fin P.parts.card)
    hg : ∀ (t : Subtype fun x => Membership.mem P.parts x), Iff (Eq (↑t).card (HAd …
    ⊢ Exists fun f => ∀ (a b : Subtype fun x => Membership.mem s x), Iff (Eq (P.pa …
  -/
  let z := fun a ↦ #P.parts * (f a).2 + g (f a).1
  /-
    case intro.intro
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    P : Finpartition s
    hP : P.IsEquipartition
    f : Equiv (Subtype fun x => Membership.mem s x) (Sigma fun t => Fin (↑t).card)
    hf : ∀ (a b : Subtype fun x => Membership.mem s x), Iff (Eq (P.part ↑a) (P.par …
    g : Equiv (Subtype fun x => Membership.mem P.parts x) (Fin P.parts.card)
    hg : ∀ (t : Subtype fun x => Membership.mem P.parts x), Iff (Eq (↑t).card (HAd …
    z : (Subtype fun x => Membership.mem s x) → Nat := fun a => HAdd.hAdd (HMul.hM …
    ⊢ Exists fun f => ∀ (a b : Subtype fun x => Membership.mem s x), Iff (Eq (P.pa …
  -/
  have gl := fun a ↦ (g (f a).1).2
  have less : ∀ a, z a < #s := fun a ↦ by
    rcases hP.card_parts_eq_average (f a).1.2 with (c | c)
    · calc
        _ < #P.parts * ((f a).2 + 1) := add_lt_add_left (gl a) _
        _ ≤ #P.parts * (#s / #P.parts) := mul_le_mul_left' (c ▸ (f a).2.2) _
        _ ≤ #P.parts * (#s / #P.parts) + #s % #P.parts := Nat.le_add_right ..
        _ = _ := Nat.div_add_mod ..
    · rw [← Nat.div_add_mod #s #P.parts]
      exact add_lt_add_of_le_of_lt (mul_le_mul_left' (by omega) _) ((hg (f a).1).mp c)
  /-
    case intro.intro
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    P : Finpartition s
    hP : P.IsEquipartition
    f : Equiv (Subtype fun x => Membership.mem s x) (Sigma fun t => Fin (↑t).card)
    hf : ∀ (a b : Subtype fun x => Membership.mem s x), Iff (Eq (P.part ↑a) (P.par …
    g : Equiv (Subtype fun x => Membership.mem P.parts x) (Fin P.parts.card)
    hg : ∀ (t : Subtype fun x => Membership.mem P.parts x), Iff (Eq (↑t).card (HAd …
    z : (Subtype fun x => Membership.mem s x) → Nat := fun a => HAdd.hAdd (HMul.hM …
    gl : ∀ (a : Subtype fun x => Membership.mem s x), LT.lt (↑(g (f a).fst)) P.par …
    less : ∀ (a : Subtype fun x => Membership.mem s x), LT.lt (z a) s.card
    ⊢ Exists fun f => ∀ (a b : Subtype fun x => Membership.mem s x), Iff (Eq (P.pa …
  -/
  let z' : s → Fin #s := fun a ↦ ⟨z a, less a⟩
  have bij : z'.Bijective := by
    refine (bijective_iff_injective_and_card z').mpr ⟨fun a b e ↦ ?_, by simp⟩
    simp_rw [z', z, Fin.mk.injEq, mul_comm #P.parts] at e
    haveI : NeZero #P.parts := ⟨((Nat.zero_le _).trans_lt (gl a)).ne'⟩
    change (#P.parts).divModEquiv.symm (_, _) = (#P.parts).divModEquiv.symm (_, _) at e
    simp only [Equiv.apply_eq_iff_eq, Prod.mk.injEq] at e
    apply_fun f
    exact Sigma.ext e.2 <| (Fin.heq_ext_iff (by rw [e.2])).mpr e.1
  /-
    case intro.intro
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    P : Finpartition s
    hP : P.IsEquipartition
    f : Equiv (Subtype fun x => Membership.mem s x) (Sigma fun t => Fin (↑t).card)
    hf : ∀ (a b : Subtype fun x => Membership.mem s x), Iff (Eq (P.part ↑a) (P.par …
    g : Equiv (Subtype fun x => Membership.mem P.parts x) (Fin P.parts.card)
    hg : ∀ (t : Subtype fun x => Membership.mem P.parts x), Iff (Eq (↑t).card (HAd …
    z : (Subtype fun x => Membership.mem s x) → Nat := fun a => HAdd.hAdd (HMul.hM …
    gl : ∀ (a : Subtype fun x => Membership.mem s x), LT.lt (↑(g (f a).fst)) P.par …
    less : ∀ (a : Subtype fun x => Membership.mem s x), LT.lt (z a) s.card
    z' : (Subtype fun x => Membership.mem s x) → Fin s.card := fun a => ⟨z a, ⋯⟩
    bij : Function.Bijective z'
    ⊢ Exists fun f => ∀ (a b : Subtype fun x => Membership.mem s x), Iff (Eq (P.pa …
  -/
  use Equiv.ofBijective _ bij
  /-
    case h
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    P : Finpartition s
    hP : P.IsEquipartition
    f : Equiv (Subtype fun x => Membership.mem s x) (Sigma fun t => Fin (↑t).card)
    hf : ∀ (a b : Subtype fun x => Membership.mem s x), Iff (Eq (P.part ↑a) (P.par …
    g : Equiv (Subtype fun x => Membership.mem P.parts x) (Fin P.parts.card)
    hg : ∀ (t : Subtype fun x => Membership.mem P.parts x), Iff (Eq (↑t).card (HAd …
    z : (Subtype fun x => Membership.mem s x) → Nat := fun a => HAdd.hAdd (HMul.hM …
    gl : ∀ (a : Subtype fun x => Membership.mem s x), LT.lt (↑(g (f a).fst)) P.par …
    less : ∀ (a : Subtype fun x => Membership.mem s x), LT.lt (z a) s.card
    z' : (Subtype fun x => Membership.mem s x) → Fin s.card := fun a => ⟨z a, ⋯⟩
    bij : Function.Bijective z'
    ⊢ ∀ (a b : Subtype fun x => Membership.mem s x), Iff (Eq (P.part ↑a) (P.part ↑ …
  -/
  intro a b
  simp_rw [z', z, Equiv.ofBijective_apply, hf a b, Nat.mul_add_mod,
    Nat.mod_eq_of_lt (gl a), Nat.mod_eq_of_lt (gl b), Fin.val_eq_val, g.apply_eq_iff_eq]


theorem bot_isEquipartition : (⊥ : Finpartition s).IsEquipartition :=
                                                    /-
                                                      α : Type u_1
                                                      inst✝ : DecidableEq α
                                                      s : Finset α
                                                      ⊢ ∀ (a : Finset α), Membership.mem (↑Bot.bot.parts) a → Or (Eq a.card 1) (Eq a …
                                                    -/
  Set.equitableOn_iff_exists_eq_eq_add_one.2 ⟨1, by simp⟩
                                                    /-
                                                      🎉 no goals
                                                    -/


theorem top_isEquipartition [Decidable (s = ⊥)] : (⊤ : Finpartition s).IsEquipartition :=
  Set.Subsingleton.isEquipartition (parts_top_subsingleton _)


theorem indiscrete_isEquipartition {hs : s ≠ ∅} : (indiscrete hs).IsEquipartition := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    hs : Ne s EmptyCollection.emptyCollection
    ⊢ (Finpartition.indiscrete hs).IsEquipartition
  -/
  rw [IsEquipartition, indiscrete_parts, coe_singleton]
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    hs : Ne s EmptyCollection.emptyCollection
    ⊢ (Singleton.singleton s).EquitableOn Finset.card
  -/
  exact Set.equitableOn_singleton s _
  /-
    🎉 no goals
  -/


