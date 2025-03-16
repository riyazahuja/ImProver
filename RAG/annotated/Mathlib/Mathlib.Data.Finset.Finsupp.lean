open scoped Classical in
/-- Finitely supported product of finsets. -/
protected def finsupp (s : Finset ι) (t : ι → Finset α) : Finset (ι →₀ α) :=
  (s.pi t).map ⟨indicator s, indicator_injective s⟩


theorem mem_finsupp_iff {t : ι → Finset α} :
    f ∈ s.finsupp t ↔ f.support ⊆ s ∧ ∀ i ∈ s, f i ∈ t i := by
  classical
  refine mem_map.trans ⟨?_, ?_⟩
  · rintro ⟨f, hf, rfl⟩
    refine ⟨support_indicator_subset _ _, fun i hi => ?_⟩
    convert mem_pi.1 hf i hi
    exact indicator_of_mem hi _
  · refine fun h => ⟨fun i _ => f i, mem_pi.2 h.2, ?_⟩
    ext i
    exact ite_eq_left_iff.2 fun hi => (not_mem_support_iff.1 fun H => hi <| h.1 H).symm


/-- When `t` is supported on `s`, `f ∈ s.finsupp t` precisely means that `f` is pointwise in `t`. -/
@[simp]
theorem mem_finsupp_iff_of_support_subset {t : ι →₀ Finset α} (ht : t.support ⊆ s) :
    f ∈ s.finsupp t ↔ ∀ i, f i ∈ t i := by
  refine
    mem_finsupp_iff.trans
      (forall_and.symm.trans <|
        forall_congr' fun i =>
          ⟨fun h => ?_, fun h =>
            ⟨fun hi => ht <| mem_support_iff.2 fun H => mem_support_iff.1 hi ?_, fun _ => h⟩⟩)
    /-
      case refine_1
      ι : Type u_1
      α : Type u_2
      inst✝ : Zero α
      s : Finset ι
      f : Finsupp ι α
      t : Finsupp ι (Finset α)
      ht : HasSubset.Subset t.support s
      i : ι
      h : And (Membership.mem f.support i → Membership.mem s i) (Membership.mem s i  …
      ⊢ Membership.mem (t i) (f i)
    -/
  · by_cases hi : i ∈ s
      /-
        case pos
        ι : Type u_1
        α : Type u_2
        inst✝ : Zero α
        s : Finset ι
        f : Finsupp ι α
        t : Finsupp ι (Finset α)
        ht : HasSubset.Subset t.support s
        i : ι
        h : And (Membership.mem f.support i → Membership.mem s i) (Membership.mem s i  …
        hi : Membership.mem s i
        ⊢ Membership.mem (t i) (f i)
      -/
    · exact h.2 hi
      /-
        🎉 no goals
      -/
      /-
        case neg
        ι : Type u_1
        α : Type u_2
        inst✝ : Zero α
        s : Finset ι
        f : Finsupp ι α
        t : Finsupp ι (Finset α)
        ht : HasSubset.Subset t.support s
        i : ι
        h : And (Membership.mem f.support i → Membership.mem s i) (Membership.mem s i  …
        hi : Not (Membership.mem s i)
        ⊢ Membership.mem (t i) (f i)
      -/
    · rw [not_mem_support_iff.1 (mt h.1 hi), not_mem_support_iff.1 fun H => hi <| ht H]
      /-
        case neg
        ι : Type u_1
        α : Type u_2
        inst✝ : Zero α
        s : Finset ι
        f : Finsupp ι α
        t : Finsupp ι (Finset α)
        ht : HasSubset.Subset t.support s
        i : ι
        h : And (Membership.mem f.support i → Membership.mem s i) (Membership.mem s i  …
        hi : Not (Membership.mem s i)
        ⊢ Membership.mem 0 0
      -/
      exact zero_mem_zero
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      ι : Type u_1
      α : Type u_2
      inst✝ : Zero α
      s : Finset ι
      f : Finsupp ι α
      t : Finsupp ι (Finset α)
      ht : HasSubset.Subset t.support s
      i : ι
      h : Membership.mem (t i) (f i)
      hi : Membership.mem f.support i
      H : Eq (t i) 0
      ⊢ Eq (f i) 0
    -/
  · rwa [H, mem_zero] at h
    /-
      🎉 no goals
    -/


@[simp]
theorem card_finsupp (s : Finset ι) (t : ι → Finset α) : #(s.finsupp t) = ∏ i ∈ s, #(t i) := by
  /-
    ι : Type u_1
    α : Type u_2
    inst✝ : Zero α
    s : Finset ι
    t : ι → Finset α
    ⊢ Eq (s.finsupp t).card (s.prod fun i => (t i).card)
  -/
  classical exact (card_map _).trans <| card_pi _ _
  /-
    🎉 no goals
  -/


/-- Given a finitely supported function `f : ι →₀ Finset α`, one can define the finset
`f.pi` of all finitely supported functions whose value at `i` is in `f i` for all `i`. -/
def pi (f : ι →₀ Finset α) : Finset (ι →₀ α) :=
  f.support.finsupp f


@[simp]
theorem mem_pi {f : ι →₀ Finset α} {g : ι →₀ α} : g ∈ f.pi ↔ ∀ i, g i ∈ f i :=
  mem_finsupp_iff_of_support_subset <| Subset.refl _


@[simp]
theorem card_pi (f : ι →₀ Finset α) : #f.pi = f.prod fun i ↦ #(f i) := by
  /-
    ι : Type u_1
    α : Type u_2
    inst✝ : Zero α
    f : Finsupp ι (Finset α)
    ⊢ Eq f.pi.card (f.prod fun i => ↑(f i).card)
  -/
  rw [pi, card_finsupp]
  /-
    ι : Type u_1
    α : Type u_2
    inst✝ : Zero α
    f : Finsupp ι (Finset α)
    ⊢ Eq (f.support.prod fun i => (f i).card) (f.prod fun i => ↑(f i).card)
  -/
  exact Finset.prod_congr rfl fun i _ => by simp only [Pi.natCast_apply, Nat.cast_id]
  /-
    🎉 no goals
  -/


