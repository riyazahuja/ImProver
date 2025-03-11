@[to_additive]
theorem smul_pi_subset [∀ i, SMul K (R i)] (r : K) (s : Set ι) (t : ∀ i, Set (R i)) :
    r • pi s t ⊆ pi s (r • t) := by
  /-
    K : Type u_1
    ι : Type u_2
    R : ι → Type u_3
    inst✝ : (i : ι) → SMul K (R i)
    r : K
    s : Set ι
    t : (i : ι) → Set (R i)
    ⊢ HasSubset.Subset (HSMul.hSMul r (s.pi t)) (s.pi (HSMul.hSMul r t))
  -/
  rintro x ⟨y, h, rfl⟩ i hi
  /-
    case intro.intro
    K : Type u_1
    ι : Type u_2
    R : ι → Type u_3
    inst✝ : (i : ι) → SMul K (R i)
    r : K
    s : Set ι
    t : (i : ι) → Set (R i)
    y : (i : ι) → R i
    h : Membership.mem (s.pi t) y
    i : ι
    hi : Membership.mem s i
    ⊢ Membership.mem (HSMul.hSMul r t i) ((fun x => HSMul.hSMul r x) y i)
  -/
  exact smul_mem_smul_set (h i hi)
  /-
    🎉 no goals
  -/

-- Porting note: Lean 4 can't synthesize `Set.mem_univ i`?

@[to_additive]
theorem smul_univ_pi [∀ i, SMul K (R i)] (r : K) (t : ∀ i, Set (R i)) :
    r • pi (univ : Set ι) t = pi (univ : Set ι) (r • t) :=
  (Subset.antisymm (smul_pi_subset _ _ _)) fun x h ↦ by
    /-
      K : Type u_1
      ι : Type u_2
      R : ι → Type u_3
      inst✝ : (i : ι) → SMul K (R i)
      r : K
      t : (i : ι) → Set (R i)
      x : (i : ι) → R i
      h : Membership.mem (Set.univ.pi (HSMul.hSMul r t)) x
      ⊢ Membership.mem (HSMul.hSMul r (Set.univ.pi t)) x
    -/
    refine ⟨fun i ↦ Classical.choose (h i <| Set.mem_univ _), fun i _ ↦ ?_, funext fun i ↦ ?_⟩
      /-
        case refine_1
        K : Type u_1
        ι : Type u_2
        R : ι → Type u_3
        inst✝ : (i : ι) → SMul K (R i)
        r : K
        t : (i : ι) → Set (R i)
        x : (i : ι) → R i
        h : Membership.mem (Set.univ.pi (HSMul.hSMul r t)) x
        i : ι
        x✝ : Membership.mem Set.univ i
        ⊢ Membership.mem (t i) ((fun i => Classical.choose ⋯) i)
      -/
    · exact (Classical.choose_spec (h i <| Set.mem_univ i)).left
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        K : Type u_1
        ι : Type u_2
        R : ι → Type u_3
        inst✝ : (i : ι) → SMul K (R i)
        r : K
        t : (i : ι) → Set (R i)
        x : (i : ι) → R i
        h : Membership.mem (Set.univ.pi (HSMul.hSMul r t)) x
        i : ι
        ⊢ Eq ((fun x => HSMul.hSMul r x) (fun i => Classical.choose ⋯) i) (x i)
      -/
    · exact (Classical.choose_spec (h i <| Set.mem_univ i)).right
      /-
        🎉 no goals
      -/


@[to_additive]
theorem smul_pi [Group K] [∀ i, MulAction K (R i)] (r : K) (S : Set ι) (t : ∀ i, Set (R i)) :
    r • S.pi t = S.pi (r • t) :=
  (Subset.antisymm (smul_pi_subset _ _ _)) fun x h ↦
    ⟨r⁻¹ • x, fun i hiS ↦ mem_smul_set_iff_inv_smul_mem.mp (h i hiS), smul_inv_smul _ _⟩


theorem smul_pi₀ [GroupWithZero K] [∀ i, MulAction K (R i)] {r : K} (S : Set ι) (t : ∀ i, Set (R i))
    (hr : r ≠ 0) : r • S.pi t = S.pi (r • t) :=
  smul_pi (Units.mk0 r hr) S t

