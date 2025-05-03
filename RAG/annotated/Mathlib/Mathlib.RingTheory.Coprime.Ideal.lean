/-- A finite family of ideals is pairwise coprime (that is, any two of them generate the whole ring)
iff when taking all the possible intersections of all but one of these ideals, the resulting family
of ideals still generate the whole ring.

For example with three ideals : `I ⊔ J = I ⊔ K = J ⊔ K = ⊤ ↔ (I ⊓ J) ⊔ (I ⊓ K) ⊔ (J ⊓ K) = ⊤`.

When ideals are all of the form `I i = R ∙ s i`, this is equivalent to the
`exists_sum_eq_one_iff_pairwise_coprime` lemma. -/
theorem iSup_iInf_eq_top_iff_pairwise {t : Finset ι} (h : t.Nonempty) (I : ι → Ideal R) :
    (⨆ i ∈ t, ⨅ (j) (_ : j ∈ t) (_ : j ≠ i), I j) = ⊤ ↔
      (t : Set ι).Pairwise fun i j => I i ⊔ I j = ⊤ := by
  /-
    ι : Type u_1
    R : Type u_2
    inst✝ : CommSemiring R
    t : Finset ι
    h : t.Nonempty
    I : ι → Ideal R
    ⊢ Iff (Eq (iSup fun i => iSup fun h => iInf fun j => iInf fun x => iInf fun x  …
  -/
  haveI : DecidableEq ι := Classical.decEq ι
  /-
    ι : Type u_1
    R : Type u_2
    inst✝ : CommSemiring R
    t : Finset ι
    h : t.Nonempty
    I : ι → Ideal R
    this : DecidableEq ι
    ⊢ Iff (Eq (iSup fun i => iSup fun h => iInf fun j => iInf fun x => iInf fun x  …
  -/
  rw [eq_top_iff_one, Submodule.mem_iSup_finset_iff_exists_sum]
  /-
    ι : Type u_1
    R : Type u_2
    inst✝ : CommSemiring R
    t : Finset ι
    h : t.Nonempty
    I : ι → Ideal R
    this : DecidableEq ι
    ⊢ Iff (Exists fun μ => Eq (t.sum fun i => ↑(μ i)) 1) ((↑t).Pairwise fun i j => …
  -/
  refine h.cons_induction ?_ ?_ <;> clear t h
    /-
      case refine_1
      ι : Type u_1
      R : Type u_2
      inst✝ : CommSemiring R
      I : ι → Ideal R
      this : DecidableEq ι
      ⊢ ∀ (a : ι), Iff (Exists fun μ => Eq ((Singleton.singleton a).sum fun i => ↑(μ …
    -/
  · simp only [Finset.sum_singleton, Finset.coe_singleton, Set.pairwise_singleton, iff_true]
    /-
      case refine_1
      ι : Type u_1
      R : Type u_2
      inst✝ : CommSemiring R
      I : ι → Ideal R
      this : DecidableEq ι
      ⊢ ∀ (a : ι), Exists fun μ => Eq (↑(μ a)) 1
    -/
    refine fun a => ⟨fun i => if h : i = a then ⟨1, ?_⟩ else 0, ?_⟩
      /-
        case refine_1.refine_1
        ι : Type u_1
        R : Type u_2
        inst✝ : CommSemiring R
        I : ι → Ideal R
        this : DecidableEq ι
        a i : ι
        h : Eq i a
        ⊢ Membership.mem (iInf fun j => iInf fun x => iInf fun x => I j) 1
      -/
    · simp [h]
      /-
        🎉 no goals
      -/
      /-
        case refine_1.refine_2
        ι : Type u_1
        R : Type u_2
        inst✝ : CommSemiring R
        I : ι → Ideal R
        this : DecidableEq ι
        a : ι
        ⊢ Eq (↑((fun i => dite (Eq i a) (fun h => ⟨1, ⋯⟩) fun h => 0) a)) 1
      -/
    · simp only [dif_pos, Submodule.coe_mk, eq_self_iff_true]
      /-
        🎉 no goals
      -/
  /-
    case refine_2
    ι : Type u_1
    R : Type u_2
    inst✝ : CommSemiring R
    I : ι → Ideal R
    this : DecidableEq ι
    ⊢ ∀ (a : ι) (s : Finset ι) (h : Not (Membership.mem s a)), s.Nonempty → Iff (E …
  -/
  intro a t hat h ih
  rw [Finset.coe_cons,
    Set.pairwise_insert_of_symmetric fun i j (h : I i ⊔ I j = ⊤) ↦ (sup_comm _ _).trans h]
  /-
    case refine_2
    ι : Type u_1
    R : Type u_2
    inst✝ : CommSemiring R
    I : ι → Ideal R
    this : DecidableEq ι
    a : ι
    t : Finset ι
    hat : Not (Membership.mem t a)
    h : t.Nonempty
    ih : Iff (Exists fun μ => Eq (t.sum fun i => ↑(μ i)) 1) ((↑t).Pairwise fun i j …
    ⊢ Iff (Exists fun μ => Eq ((Finset.cons a t hat).sum fun i => ↑(μ i)) 1) (And  …
  -/
  constructor
    /-
      case refine_2.mp
      ι : Type u_1
      R : Type u_2
      inst✝ : CommSemiring R
      I : ι → Ideal R
      this : DecidableEq ι
      a : ι
      t : Finset ι
      hat : Not (Membership.mem t a)
      h : t.Nonempty
      ih : Iff (Exists fun μ => Eq (t.sum fun i => ↑(μ i)) 1) ((↑t).Pairwise fun i j …
      ⊢ (Exists fun μ => Eq ((Finset.cons a t hat).sum fun i => ↑(μ i)) 1) → And ((↑ …
    -/
  · rintro ⟨μ, hμ⟩
    /-
      case refine_2.mp.intro
      ι : Type u_1
      R : Type u_2
      inst✝ : CommSemiring R
      I : ι → Ideal R
      this : DecidableEq ι
      a : ι
      t : Finset ι
      hat : Not (Membership.mem t a)
      h : t.Nonempty
      ih : Iff (Exists fun μ => Eq (t.sum fun i => ↑(μ i)) 1) ((↑t).Pairwise fun i j …
      μ : (i : ι) → Subtype fun x => Membership.mem (iInf fun j => iInf fun x => iIn …
      hμ : Eq ((Finset.cons a t hat).sum fun i => ↑(μ i)) 1
      ⊢ And ((↑t).Pairwise fun i j => Eq (Max.max (I i) (I j)) Top.top) (∀ (b : ι),  …
    -/
    rw [Finset.sum_cons] at hμ
    -- Porting note: `refine` yields goals in a different order than in lean3.
    /-
      case refine_2.mp.intro
      ι : Type u_1
      R : Type u_2
      inst✝ : CommSemiring R
      I : ι → Ideal R
      this : DecidableEq ι
      a : ι
      t : Finset ι
      hat : Not (Membership.mem t a)
      h : t.Nonempty
      ih : Iff (Exists fun μ => Eq (t.sum fun i => ↑(μ i)) 1) ((↑t).Pairwise fun i j …
      μ : (i : ι) → Subtype fun x => Membership.mem (iInf fun j => iInf fun x => iIn …
      hμ : Eq (HAdd.hAdd (↑(μ a)) (t.sum fun x => ↑(μ x))) 1
      ⊢ And ((↑t).Pairwise fun i j => Eq (Max.max (I i) (I j)) Top.top) (∀ (b : ι),  …
    -/
    refine ⟨ih.mp ⟨Pi.single h.choose ⟨μ a, ?a1⟩ + fun i => ⟨μ i, ?a2⟩, ?a3⟩, fun b hb ab => ?a4⟩
    case a1 =>
      have := Submodule.coe_mem (μ a)
      rw [mem_iInf] at this ⊢
      --for some reason `simp only [mem_iInf]` times out
      intro i
      specialize this i
      rw [mem_iInf, mem_iInf] at this ⊢
      intro hi _
      apply this (Finset.subset_cons _ hi)
      rintro rfl
      exact hat hi
    case a2 =>
      have := Submodule.coe_mem (μ i)
      simp only [mem_iInf] at this ⊢
      intro j hj ij
      exact this _ (Finset.subset_cons _ hj) ij
    case a3 =>
      rw [← @if_pos _ _ h.choose_spec R (μ a) 0, ← Finset.sum_pi_single', ← Finset.sum_add_distrib]
        at hμ
      convert hμ
      rename_i i _
      rw [Pi.add_apply, Submodule.coe_add, Submodule.coe_mk]
      by_cases hi : i = h.choose
      · rw [hi, Pi.single_eq_same, Pi.single_eq_same, Submodule.coe_mk]
      · rw [Pi.single_eq_of_ne hi, Pi.single_eq_of_ne hi, Submodule.coe_zero]
    case a4 =>
      rw [eq_top_iff_one, Submodule.mem_sup]
      rw [add_comm] at hμ
      refine ⟨_, ?_, _, ?_, hμ⟩
      · refine sum_mem _ fun x hx => ?_
        have := Submodule.coe_mem (μ x)
        simp only [mem_iInf] at this
        apply this _ (Finset.mem_cons_self _ _)
        rintro rfl
        exact hat hx
      · have := Submodule.coe_mem (μ a)
        simp only [mem_iInf] at this
        exact this _ (Finset.subset_cons _ hb) ab.symm
    /-
      case refine_2.mpr
      ι : Type u_1
      R : Type u_2
      inst✝ : CommSemiring R
      I : ι → Ideal R
      this : DecidableEq ι
      a : ι
      t : Finset ι
      hat : Not (Membership.mem t a)
      h : t.Nonempty
      ih : Iff (Exists fun μ => Eq (t.sum fun i => ↑(μ i)) 1) ((↑t).Pairwise fun i j …
      ⊢ And ((↑t).Pairwise fun i j => Eq (Max.max (I i) (I j)) Top.top) (∀ (b : ι),  …
    -/
  · rintro ⟨hs, Hb⟩
    /-
      case refine_2.mpr.intro
      ι : Type u_1
      R : Type u_2
      inst✝ : CommSemiring R
      I : ι → Ideal R
      this : DecidableEq ι
      a : ι
      t : Finset ι
      hat : Not (Membership.mem t a)
      h : t.Nonempty
      ih : Iff (Exists fun μ => Eq (t.sum fun i => ↑(μ i)) 1) ((↑t).Pairwise fun i j …
      hs : (↑t).Pairwise fun i j => Eq (Max.max (I i) (I j)) Top.top
      Hb : ∀ (b : ι), Membership.mem (↑t) b → Ne a b → Eq (Max.max (I a) (I b)) Top. …
      ⊢ Exists fun μ => Eq ((Finset.cons a t hat).sum fun i => ↑(μ i)) 1
    -/
    obtain ⟨μ, hμ⟩ := ih.mpr hs
    /-
      case refine_2.mpr.intro.intro
      ι : Type u_1
      R : Type u_2
      inst✝ : CommSemiring R
      I : ι → Ideal R
      this : DecidableEq ι
      a : ι
      t : Finset ι
      hat : Not (Membership.mem t a)
      h : t.Nonempty
      ih : Iff (Exists fun μ => Eq (t.sum fun i => ↑(μ i)) 1) ((↑t).Pairwise fun i j …
      hs : (↑t).Pairwise fun i j => Eq (Max.max (I i) (I j)) Top.top
      Hb : ∀ (b : ι), Membership.mem (↑t) b → Ne a b → Eq (Max.max (I a) (I b)) Top. …
      μ : (i : ι) → Subtype fun x => Membership.mem (iInf fun j => iInf fun x => iIn …
      hμ : Eq (t.sum fun i => ↑(μ i)) 1
      ⊢ Exists fun μ => Eq ((Finset.cons a t hat).sum fun i => ↑(μ i)) 1
    -/
    have := sup_iInf_eq_top fun b hb => Hb b hb (ne_of_mem_of_not_mem hb hat).symm
    /-
      case refine_2.mpr.intro.intro
      ι : Type u_1
      R : Type u_2
      inst✝ : CommSemiring R
      I : ι → Ideal R
      this✝ : DecidableEq ι
      a : ι
      t : Finset ι
      hat : Not (Membership.mem t a)
      h : t.Nonempty
      ih : Iff (Exists fun μ => Eq (t.sum fun i => ↑(μ i)) 1) ((↑t).Pairwise fun i j …
      hs : (↑t).Pairwise fun i j => Eq (Max.max (I i) (I j)) Top.top
      Hb : ∀ (b : ι), Membership.mem (↑t) b → Ne a b → Eq (Max.max (I a) (I b)) Top. …
      μ : (i : ι) → Subtype fun x => Membership.mem (iInf fun j => iInf fun x => iIn …
      hμ : Eq (t.sum fun i => ↑(μ i)) 1
      this : Eq (Max.max (I a) (iInf fun i => iInf fun h => I i)) Top.top
      ⊢ Exists fun μ => Eq ((Finset.cons a t hat).sum fun i => ↑(μ i)) 1
    -/
    rw [eq_top_iff_one, Submodule.mem_sup] at this
    /-
      case refine_2.mpr.intro.intro
      ι : Type u_1
      R : Type u_2
      inst✝ : CommSemiring R
      I : ι → Ideal R
      this✝ : DecidableEq ι
      a : ι
      t : Finset ι
      hat : Not (Membership.mem t a)
      h : t.Nonempty
      ih : Iff (Exists fun μ => Eq (t.sum fun i => ↑(μ i)) 1) ((↑t).Pairwise fun i j …
      hs : (↑t).Pairwise fun i j => Eq (Max.max (I i) (I j)) Top.top
      Hb : ∀ (b : ι), Membership.mem (↑t) b → Ne a b → Eq (Max.max (I a) (I b)) Top. …
      μ : (i : ι) → Subtype fun x => Membership.mem (iInf fun j => iInf fun x => iIn …
      hμ : Eq (t.sum fun i => ↑(μ i)) 1
      this : Exists fun y => And (Membership.mem (I a) y) (Exists fun z => And (Memb …
      ⊢ Exists fun μ => Eq ((Finset.cons a t hat).sum fun i => ↑(μ i)) 1
    -/
    obtain ⟨u, hu, v, hv, huv⟩ := this
    /-
      case refine_2.mpr.intro.intro.intro.intro.intro.intro
      ι : Type u_1
      R : Type u_2
      inst✝ : CommSemiring R
      I : ι → Ideal R
      this : DecidableEq ι
      a : ι
      t : Finset ι
      hat : Not (Membership.mem t a)
      h : t.Nonempty
      ih : Iff (Exists fun μ => Eq (t.sum fun i => ↑(μ i)) 1) ((↑t).Pairwise fun i j …
      hs : (↑t).Pairwise fun i j => Eq (Max.max (I i) (I j)) Top.top
      Hb : ∀ (b : ι), Membership.mem (↑t) b → Ne a b → Eq (Max.max (I a) (I b)) Top. …
      μ : (i : ι) → Subtype fun x => Membership.mem (iInf fun j => iInf fun x => iIn …
      hμ : Eq (t.sum fun i => ↑(μ i)) 1
      u : R
      hu : Membership.mem (I a) u
      v : R
      hv : Membership.mem (iInf fun i => iInf fun h => I i) v
      huv : Eq (HAdd.hAdd u v) 1
      ⊢ Exists fun μ => Eq ((Finset.cons a t hat).sum fun i => ↑(μ i)) 1
    -/
    refine ⟨fun i => if hi : i = a then ⟨v, ?_⟩ else ⟨u * μ i, ?_⟩, ?_⟩
      /-
        case refine_2.mpr.intro.intro.intro.intro.intro.intro.refine_1
        ι : Type u_1
        R : Type u_2
        inst✝ : CommSemiring R
        I : ι → Ideal R
        this : DecidableEq ι
        a : ι
        t : Finset ι
        hat : Not (Membership.mem t a)
        h : t.Nonempty
        ih : Iff (Exists fun μ => Eq (t.sum fun i => ↑(μ i)) 1) ((↑t).Pairwise fun i j …
        hs : (↑t).Pairwise fun i j => Eq (Max.max (I i) (I j)) Top.top
        Hb : ∀ (b : ι), Membership.mem (↑t) b → Ne a b → Eq (Max.max (I a) (I b)) Top. …
        μ : (i : ι) → Subtype fun x => Membership.mem (iInf fun j => iInf fun x => iIn …
        hμ : Eq (t.sum fun i => ↑(μ i)) 1
        u : R
        hu : Membership.mem (I a) u
        v : R
        hv : Membership.mem (iInf fun i => iInf fun h => I i) v
        huv : Eq (HAdd.hAdd u v) 1
        i : ι
        hi : Eq i a
        ⊢ Membership.mem (iInf fun j => iInf fun x => iInf fun x => I j) v
      -/
    · simp only [mem_iInf] at hv ⊢
      /-
        case refine_2.mpr.intro.intro.intro.intro.intro.intro.refine_1
        ι : Type u_1
        R : Type u_2
        inst✝ : CommSemiring R
        I : ι → Ideal R
        this : DecidableEq ι
        a : ι
        t : Finset ι
        hat : Not (Membership.mem t a)
        h : t.Nonempty
        ih : Iff (Exists fun μ => Eq (t.sum fun i => ↑(μ i)) 1) ((↑t).Pairwise fun i j …
        hs : (↑t).Pairwise fun i j => Eq (Max.max (I i) (I j)) Top.top
        Hb : ∀ (b : ι), Membership.mem (↑t) b → Ne a b → Eq (Max.max (I a) (I b)) Top. …
        μ : (i : ι) → Subtype fun x => Membership.mem (iInf fun j => iInf fun x => iIn …
        hμ : Eq (t.sum fun i => ↑(μ i)) 1
        u : R
        hu : Membership.mem (I a) u
        v : R
        huv : Eq (HAdd.hAdd u v) 1
        i : ι
        hi : Eq i a
        hv : ∀ (i : ι), Membership.mem t i → Membership.mem (I i) v
        ⊢ ∀ (i_1 : ι), Membership.mem (Finset.cons a t hat) i_1 → Ne i_1 i → Membershi …
      -/
      intro j hj ij
      /-
        case refine_2.mpr.intro.intro.intro.intro.intro.intro.refine_1
        ι : Type u_1
        R : Type u_2
        inst✝ : CommSemiring R
        I : ι → Ideal R
        this : DecidableEq ι
        a : ι
        t : Finset ι
        hat : Not (Membership.mem t a)
        h : t.Nonempty
        ih : Iff (Exists fun μ => Eq (t.sum fun i => ↑(μ i)) 1) ((↑t).Pairwise fun i j …
        hs : (↑t).Pairwise fun i j => Eq (Max.max (I i) (I j)) Top.top
        Hb : ∀ (b : ι), Membership.mem (↑t) b → Ne a b → Eq (Max.max (I a) (I b)) Top. …
        μ : (i : ι) → Subtype fun x => Membership.mem (iInf fun j => iInf fun x => iIn …
        hμ : Eq (t.sum fun i => ↑(μ i)) 1
        u : R
        hu : Membership.mem (I a) u
        v : R
        huv : Eq (HAdd.hAdd u v) 1
        i : ι
        hi : Eq i a
        hv : ∀ (i : ι), Membership.mem t i → Membership.mem (I i) v
        j : ι
        hj : Membership.mem (Finset.cons a t hat) j
        ij : Ne j i
        ⊢ Membership.mem (I j) v
      -/
      rw [Finset.mem_cons, ← hi] at hj
      /-
        case refine_2.mpr.intro.intro.intro.intro.intro.intro.refine_1
        ι : Type u_1
        R : Type u_2
        inst✝ : CommSemiring R
        I : ι → Ideal R
        this : DecidableEq ι
        a : ι
        t : Finset ι
        hat : Not (Membership.mem t a)
        h : t.Nonempty
        ih : Iff (Exists fun μ => Eq (t.sum fun i => ↑(μ i)) 1) ((↑t).Pairwise fun i j …
        hs : (↑t).Pairwise fun i j => Eq (Max.max (I i) (I j)) Top.top
        Hb : ∀ (b : ι), Membership.mem (↑t) b → Ne a b → Eq (Max.max (I a) (I b)) Top. …
        μ : (i : ι) → Subtype fun x => Membership.mem (iInf fun j => iInf fun x => iIn …
        hμ : Eq (t.sum fun i => ↑(μ i)) 1
        u : R
        hu : Membership.mem (I a) u
        v : R
        huv : Eq (HAdd.hAdd u v) 1
        i : ι
        hi : Eq i a
        hv : ∀ (i : ι), Membership.mem t i → Membership.mem (I i) v
        j : ι
        hj : Or (Eq j i) (Membership.mem t j)
        ij : Ne j i
        ⊢ Membership.mem (I j) v
      -/
      exact hv _ (hj.resolve_left ij)
      /-
        🎉 no goals
      -/
      /-
        case refine_2.mpr.intro.intro.intro.intro.intro.intro.refine_2
        ι : Type u_1
        R : Type u_2
        inst✝ : CommSemiring R
        I : ι → Ideal R
        this : DecidableEq ι
        a : ι
        t : Finset ι
        hat : Not (Membership.mem t a)
        h : t.Nonempty
        ih : Iff (Exists fun μ => Eq (t.sum fun i => ↑(μ i)) 1) ((↑t).Pairwise fun i j …
        hs : (↑t).Pairwise fun i j => Eq (Max.max (I i) (I j)) Top.top
        Hb : ∀ (b : ι), Membership.mem (↑t) b → Ne a b → Eq (Max.max (I a) (I b)) Top. …
        μ : (i : ι) → Subtype fun x => Membership.mem (iInf fun j => iInf fun x => iIn …
        hμ : Eq (t.sum fun i => ↑(μ i)) 1
        u : R
        hu : Membership.mem (I a) u
        v : R
        hv : Membership.mem (iInf fun i => iInf fun h => I i) v
        huv : Eq (HAdd.hAdd u v) 1
        i : ι
        hi : Not (Eq i a)
        ⊢ Membership.mem (iInf fun j => iInf fun x => iInf fun x => I j) (HMul.hMul u  …
      -/
    · have := Submodule.coe_mem (μ i)
      /-
        case refine_2.mpr.intro.intro.intro.intro.intro.intro.refine_2
        ι : Type u_1
        R : Type u_2
        inst✝ : CommSemiring R
        I : ι → Ideal R
        this✝ : DecidableEq ι
        a : ι
        t : Finset ι
        hat : Not (Membership.mem t a)
        h : t.Nonempty
        ih : Iff (Exists fun μ => Eq (t.sum fun i => ↑(μ i)) 1) ((↑t).Pairwise fun i j …
        hs : (↑t).Pairwise fun i j => Eq (Max.max (I i) (I j)) Top.top
        Hb : ∀ (b : ι), Membership.mem (↑t) b → Ne a b → Eq (Max.max (I a) (I b)) Top. …
        μ : (i : ι) → Subtype fun x => Membership.mem (iInf fun j => iInf fun x => iIn …
        hμ : Eq (t.sum fun i => ↑(μ i)) 1
        u : R
        hu : Membership.mem (I a) u
        v : R
        hv : Membership.mem (iInf fun i => iInf fun h => I i) v
        huv : Eq (HAdd.hAdd u v) 1
        i : ι
        hi : Not (Eq i a)
        this : Membership.mem (iInf fun j => iInf fun x => iInf fun x => I j) ↑(μ i)
        ⊢ Membership.mem (iInf fun j => iInf fun x => iInf fun x => I j) (HMul.hMul u  …
      -/
      simp only [mem_iInf] at this ⊢
      /-
        case refine_2.mpr.intro.intro.intro.intro.intro.intro.refine_2
        ι : Type u_1
        R : Type u_2
        inst✝ : CommSemiring R
        I : ι → Ideal R
        this✝ : DecidableEq ι
        a : ι
        t : Finset ι
        hat : Not (Membership.mem t a)
        h : t.Nonempty
        ih : Iff (Exists fun μ => Eq (t.sum fun i => ↑(μ i)) 1) ((↑t).Pairwise fun i j …
        hs : (↑t).Pairwise fun i j => Eq (Max.max (I i) (I j)) Top.top
        Hb : ∀ (b : ι), Membership.mem (↑t) b → Ne a b → Eq (Max.max (I a) (I b)) Top. …
        μ : (i : ι) → Subtype fun x => Membership.mem (iInf fun j => iInf fun x => iIn …
        hμ : Eq (t.sum fun i => ↑(μ i)) 1
        u : R
        hu : Membership.mem (I a) u
        v : R
        hv : Membership.mem (iInf fun i => iInf fun h => I i) v
        huv : Eq (HAdd.hAdd u v) 1
        i : ι
        hi : Not (Eq i a)
        this : ∀ (i_1 : ι), Membership.mem t i_1 → Ne i_1 i → Membership.mem (I i_1) ↑ …
        ⊢ ∀ (i_1 : ι), Membership.mem (Finset.cons a t hat) i_1 → Ne i_1 i → Membershi …
      -/
      intro j hj ij
      /-
        case refine_2.mpr.intro.intro.intro.intro.intro.intro.refine_2
        ι : Type u_1
        R : Type u_2
        inst✝ : CommSemiring R
        I : ι → Ideal R
        this✝ : DecidableEq ι
        a : ι
        t : Finset ι
        hat : Not (Membership.mem t a)
        h : t.Nonempty
        ih : Iff (Exists fun μ => Eq (t.sum fun i => ↑(μ i)) 1) ((↑t).Pairwise fun i j …
        hs : (↑t).Pairwise fun i j => Eq (Max.max (I i) (I j)) Top.top
        Hb : ∀ (b : ι), Membership.mem (↑t) b → Ne a b → Eq (Max.max (I a) (I b)) Top. …
        μ : (i : ι) → Subtype fun x => Membership.mem (iInf fun j => iInf fun x => iIn …
        hμ : Eq (t.sum fun i => ↑(μ i)) 1
        u : R
        hu : Membership.mem (I a) u
        v : R
        hv : Membership.mem (iInf fun i => iInf fun h => I i) v
        huv : Eq (HAdd.hAdd u v) 1
        i : ι
        hi : Not (Eq i a)
        this : ∀ (i_1 : ι), Membership.mem t i_1 → Ne i_1 i → Membership.mem (I i_1) ↑ …
        j : ι
        hj : Membership.mem (Finset.cons a t hat) j
        ij : Ne j i
        ⊢ Membership.mem (I j) (HMul.hMul u ↑(μ i))
      -/
      rcases Finset.mem_cons.mp hj with (rfl | hj)
        /-
          case refine_2.mpr.intro.intro.intro.intro.intro.intro.refine_2.inl
          ι : Type u_1
          R : Type u_2
          inst✝ : CommSemiring R
          I : ι → Ideal R
          this✝ : DecidableEq ι
          t : Finset ι
          h : t.Nonempty
          ih : Iff (Exists fun μ => Eq (t.sum fun i => ↑(μ i)) 1) ((↑t).Pairwise fun i j …
          hs : (↑t).Pairwise fun i j => Eq (Max.max (I i) (I j)) Top.top
          μ : (i : ι) → Subtype fun x => Membership.mem (iInf fun j => iInf fun x => iIn …
          hμ : Eq (t.sum fun i => ↑(μ i)) 1
          u v : R
          hv : Membership.mem (iInf fun i => iInf fun h => I i) v
          huv : Eq (HAdd.hAdd u v) 1
          i : ι
          this : ∀ (i_1 : ι), Membership.mem t i_1 → Ne i_1 i → Membership.mem (I i_1) ↑ …
          j : ι
          ij : Ne j i
          hat : Not (Membership.mem t j)
          Hb : ∀ (b : ι), Membership.mem (↑t) b → Ne j b → Eq (Max.max (I j) (I b)) Top. …
          hu : Membership.mem (I j) u
          hi : Not (Eq i j)
          hj : Membership.mem (Finset.cons j t hat) j
          ⊢ Membership.mem (I j) (HMul.hMul u ↑(μ i))
        -/
      · exact mul_mem_right _ _ hu
        /-
          🎉 no goals
        -/
        /-
          case refine_2.mpr.intro.intro.intro.intro.intro.intro.refine_2.inr
          ι : Type u_1
          R : Type u_2
          inst✝ : CommSemiring R
          I : ι → Ideal R
          this✝ : DecidableEq ι
          a : ι
          t : Finset ι
          hat : Not (Membership.mem t a)
          h : t.Nonempty
          ih : Iff (Exists fun μ => Eq (t.sum fun i => ↑(μ i)) 1) ((↑t).Pairwise fun i j …
          hs : (↑t).Pairwise fun i j => Eq (Max.max (I i) (I j)) Top.top
          Hb : ∀ (b : ι), Membership.mem (↑t) b → Ne a b → Eq (Max.max (I a) (I b)) Top. …
          μ : (i : ι) → Subtype fun x => Membership.mem (iInf fun j => iInf fun x => iIn …
          hμ : Eq (t.sum fun i => ↑(μ i)) 1
          u : R
          hu : Membership.mem (I a) u
          v : R
          hv : Membership.mem (iInf fun i => iInf fun h => I i) v
          huv : Eq (HAdd.hAdd u v) 1
          i : ι
          hi : Not (Eq i a)
          this : ∀ (i_1 : ι), Membership.mem t i_1 → Ne i_1 i → Membership.mem (I i_1) ↑ …
          j : ι
          hj✝ : Membership.mem (Finset.cons a t hat) j
          ij : Ne j i
          hj : Membership.mem t j
          ⊢ Membership.mem (I j) (HMul.hMul u ↑(μ i))
        -/
      · exact mul_mem_left _ _ (this _ hj ij)
        /-
          🎉 no goals
        -/
      /-
        case refine_2.mpr.intro.intro.intro.intro.intro.intro.refine_3
        ι : Type u_1
        R : Type u_2
        inst✝ : CommSemiring R
        I : ι → Ideal R
        this : DecidableEq ι
        a : ι
        t : Finset ι
        hat : Not (Membership.mem t a)
        h : t.Nonempty
        ih : Iff (Exists fun μ => Eq (t.sum fun i => ↑(μ i)) 1) ((↑t).Pairwise fun i j …
        hs : (↑t).Pairwise fun i j => Eq (Max.max (I i) (I j)) Top.top
        Hb : ∀ (b : ι), Membership.mem (↑t) b → Ne a b → Eq (Max.max (I a) (I b)) Top. …
        μ : (i : ι) → Subtype fun x => Membership.mem (iInf fun j => iInf fun x => iIn …
        hμ : Eq (t.sum fun i => ↑(μ i)) 1
        u : R
        hu : Membership.mem (I a) u
        v : R
        hv : Membership.mem (iInf fun i => iInf fun h => I i) v
        huv : Eq (HAdd.hAdd u v) 1
        ⊢ Eq ((Finset.cons a t hat).sum fun i => ↑((fun i => dite (Eq i a) (fun hi =>  …
      -/
    · dsimp only
      /-
        case refine_2.mpr.intro.intro.intro.intro.intro.intro.refine_3
        ι : Type u_1
        R : Type u_2
        inst✝ : CommSemiring R
        I : ι → Ideal R
        this : DecidableEq ι
        a : ι
        t : Finset ι
        hat : Not (Membership.mem t a)
        h : t.Nonempty
        ih : Iff (Exists fun μ => Eq (t.sum fun i => ↑(μ i)) 1) ((↑t).Pairwise fun i j …
        hs : (↑t).Pairwise fun i j => Eq (Max.max (I i) (I j)) Top.top
        Hb : ∀ (b : ι), Membership.mem (↑t) b → Ne a b → Eq (Max.max (I a) (I b)) Top. …
        μ : (i : ι) → Subtype fun x => Membership.mem (iInf fun j => iInf fun x => iIn …
        hμ : Eq (t.sum fun i => ↑(μ i)) 1
        u : R
        hu : Membership.mem (I a) u
        v : R
        hv : Membership.mem (iInf fun i => iInf fun h => I i) v
        huv : Eq (HAdd.hAdd u v) 1
        ⊢ Eq ((Finset.cons a t hat).sum fun i => ↑(dite (Eq i a) (fun hi => ⟨v, ⋯⟩) fu …
      -/
      rw [Finset.sum_cons, dif_pos rfl, add_comm]
      /-
        case refine_2.mpr.intro.intro.intro.intro.intro.intro.refine_3
        ι : Type u_1
        R : Type u_2
        inst✝ : CommSemiring R
        I : ι → Ideal R
        this : DecidableEq ι
        a : ι
        t : Finset ι
        hat : Not (Membership.mem t a)
        h : t.Nonempty
        ih : Iff (Exists fun μ => Eq (t.sum fun i => ↑(μ i)) 1) ((↑t).Pairwise fun i j …
        hs : (↑t).Pairwise fun i j => Eq (Max.max (I i) (I j)) Top.top
        Hb : ∀ (b : ι), Membership.mem (↑t) b → Ne a b → Eq (Max.max (I a) (I b)) Top. …
        μ : (i : ι) → Subtype fun x => Membership.mem (iInf fun j => iInf fun x => iIn …
        hμ : Eq (t.sum fun i => ↑(μ i)) 1
        u : R
        hu : Membership.mem (I a) u
        v : R
        hv : Membership.mem (iInf fun i => iInf fun h => I i) v
        huv : Eq (HAdd.hAdd u v) 1
        ⊢ Eq (HAdd.hAdd (t.sum fun x => ↑(dite (Eq x a) (fun hi => ⟨v, ⋯⟩) fun hi => ⟨ …
      -/
      rw [← mul_one u] at huv
      /-
        case refine_2.mpr.intro.intro.intro.intro.intro.intro.refine_3
        ι : Type u_1
        R : Type u_2
        inst✝ : CommSemiring R
        I : ι → Ideal R
        this : DecidableEq ι
        a : ι
        t : Finset ι
        hat : Not (Membership.mem t a)
        h : t.Nonempty
        ih : Iff (Exists fun μ => Eq (t.sum fun i => ↑(μ i)) 1) ((↑t).Pairwise fun i j …
        hs : (↑t).Pairwise fun i j => Eq (Max.max (I i) (I j)) Top.top
        Hb : ∀ (b : ι), Membership.mem (↑t) b → Ne a b → Eq (Max.max (I a) (I b)) Top. …
        μ : (i : ι) → Subtype fun x => Membership.mem (iInf fun j => iInf fun x => iIn …
        hμ : Eq (t.sum fun i => ↑(μ i)) 1
        u : R
        hu : Membership.mem (I a) u
        v : R
        hv : Membership.mem (iInf fun i => iInf fun h => I i) v
        huv : Eq (HAdd.hAdd (HMul.hMul u 1) v) 1
        ⊢ Eq (HAdd.hAdd (t.sum fun x => ↑(dite (Eq x a) (fun hi => ⟨v, ⋯⟩) fun hi => ⟨ …
      -/
      rw [← huv, ← hμ, Finset.mul_sum]
      /-
        case refine_2.mpr.intro.intro.intro.intro.intro.intro.refine_3
        ι : Type u_1
        R : Type u_2
        inst✝ : CommSemiring R
        I : ι → Ideal R
        this : DecidableEq ι
        a : ι
        t : Finset ι
        hat : Not (Membership.mem t a)
        h : t.Nonempty
        ih : Iff (Exists fun μ => Eq (t.sum fun i => ↑(μ i)) 1) ((↑t).Pairwise fun i j …
        hs : (↑t).Pairwise fun i j => Eq (Max.max (I i) (I j)) Top.top
        Hb : ∀ (b : ι), Membership.mem (↑t) b → Ne a b → Eq (Max.max (I a) (I b)) Top. …
        μ : (i : ι) → Subtype fun x => Membership.mem (iInf fun j => iInf fun x => iIn …
        hμ : Eq (t.sum fun i => ↑(μ i)) 1
        u : R
        hu : Membership.mem (I a) u
        v : R
        hv : Membership.mem (iInf fun i => iInf fun h => I i) v
        huv : Eq (HAdd.hAdd (HMul.hMul u 1) v) 1
        ⊢ Eq (HAdd.hAdd (t.sum fun x => ↑(dite (Eq x a) (fun hi => ⟨v, ⋯⟩) fun hi => ⟨ …
      -/
      congr 1
      /-
        case refine_2.mpr.intro.intro.intro.intro.intro.intro.refine_3.e_a
        ι : Type u_1
        R : Type u_2
        inst✝ : CommSemiring R
        I : ι → Ideal R
        this : DecidableEq ι
        a : ι
        t : Finset ι
        hat : Not (Membership.mem t a)
        h : t.Nonempty
        ih : Iff (Exists fun μ => Eq (t.sum fun i => ↑(μ i)) 1) ((↑t).Pairwise fun i j …
        hs : (↑t).Pairwise fun i j => Eq (Max.max (I i) (I j)) Top.top
        Hb : ∀ (b : ι), Membership.mem (↑t) b → Ne a b → Eq (Max.max (I a) (I b)) Top. …
        μ : (i : ι) → Subtype fun x => Membership.mem (iInf fun j => iInf fun x => iIn …
        hμ : Eq (t.sum fun i => ↑(μ i)) 1
        u : R
        hu : Membership.mem (I a) u
        v : R
        hv : Membership.mem (iInf fun i => iInf fun h => I i) v
        huv : Eq (HAdd.hAdd (HMul.hMul u 1) v) 1
        ⊢ Eq (t.sum fun x => ↑(dite (Eq x a) (fun hi => ⟨v, ⋯⟩) fun hi => ⟨HMul.hMul u …
      -/
      apply Finset.sum_congr rfl
      /-
        case refine_2.mpr.intro.intro.intro.intro.intro.intro.refine_3.e_a
        ι : Type u_1
        R : Type u_2
        inst✝ : CommSemiring R
        I : ι → Ideal R
        this : DecidableEq ι
        a : ι
        t : Finset ι
        hat : Not (Membership.mem t a)
        h : t.Nonempty
        ih : Iff (Exists fun μ => Eq (t.sum fun i => ↑(μ i)) 1) ((↑t).Pairwise fun i j …
        hs : (↑t).Pairwise fun i j => Eq (Max.max (I i) (I j)) Top.top
        Hb : ∀ (b : ι), Membership.mem (↑t) b → Ne a b → Eq (Max.max (I a) (I b)) Top. …
        μ : (i : ι) → Subtype fun x => Membership.mem (iInf fun j => iInf fun x => iIn …
        hμ : Eq (t.sum fun i => ↑(μ i)) 1
        u : R
        hu : Membership.mem (I a) u
        v : R
        hv : Membership.mem (iInf fun i => iInf fun h => I i) v
        huv : Eq (HAdd.hAdd (HMul.hMul u 1) v) 1
        ⊢ ∀ (x : ι), Membership.mem t x → Eq (↑(dite (Eq x a) (fun hi => ⟨v, ⋯⟩) fun h …
      -/
      intro j hj
      /-
        case refine_2.mpr.intro.intro.intro.intro.intro.intro.refine_3.e_a
        ι : Type u_1
        R : Type u_2
        inst✝ : CommSemiring R
        I : ι → Ideal R
        this : DecidableEq ι
        a : ι
        t : Finset ι
        hat : Not (Membership.mem t a)
        h : t.Nonempty
        ih : Iff (Exists fun μ => Eq (t.sum fun i => ↑(μ i)) 1) ((↑t).Pairwise fun i j …
        hs : (↑t).Pairwise fun i j => Eq (Max.max (I i) (I j)) Top.top
        Hb : ∀ (b : ι), Membership.mem (↑t) b → Ne a b → Eq (Max.max (I a) (I b)) Top. …
        μ : (i : ι) → Subtype fun x => Membership.mem (iInf fun j => iInf fun x => iIn …
        hμ : Eq (t.sum fun i => ↑(μ i)) 1
        u : R
        hu : Membership.mem (I a) u
        v : R
        hv : Membership.mem (iInf fun i => iInf fun h => I i) v
        huv : Eq (HAdd.hAdd (HMul.hMul u 1) v) 1
        j : ι
        hj : Membership.mem t j
        ⊢ Eq (↑(dite (Eq j a) (fun hi => ⟨v, ⋯⟩) fun hi => ⟨HMul.hMul u ↑(μ j), ⋯⟩)) ( …
      -/
      rw [dif_neg]
      /-
        case refine_2.mpr.intro.intro.intro.intro.intro.intro.refine_3.e_a.hnc
        ι : Type u_1
        R : Type u_2
        inst✝ : CommSemiring R
        I : ι → Ideal R
        this : DecidableEq ι
        a : ι
        t : Finset ι
        hat : Not (Membership.mem t a)
        h : t.Nonempty
        ih : Iff (Exists fun μ => Eq (t.sum fun i => ↑(μ i)) 1) ((↑t).Pairwise fun i j …
        hs : (↑t).Pairwise fun i j => Eq (Max.max (I i) (I j)) Top.top
        Hb : ∀ (b : ι), Membership.mem (↑t) b → Ne a b → Eq (Max.max (I a) (I b)) Top. …
        μ : (i : ι) → Subtype fun x => Membership.mem (iInf fun j => iInf fun x => iIn …
        hμ : Eq (t.sum fun i => ↑(μ i)) 1
        u : R
        hu : Membership.mem (I a) u
        v : R
        hv : Membership.mem (iInf fun i => iInf fun h => I i) v
        huv : Eq (HAdd.hAdd (HMul.hMul u 1) v) 1
        j : ι
        hj : Membership.mem t j
        ⊢ Not (Eq j a)
      -/
      rintro rfl
      /-
        case refine_2.mpr.intro.intro.intro.intro.intro.intro.refine_3.e_a.hnc
        ι : Type u_1
        R : Type u_2
        inst✝ : CommSemiring R
        I : ι → Ideal R
        this : DecidableEq ι
        t : Finset ι
        h : t.Nonempty
        ih : Iff (Exists fun μ => Eq (t.sum fun i => ↑(μ i)) 1) ((↑t).Pairwise fun i j …
        hs : (↑t).Pairwise fun i j => Eq (Max.max (I i) (I j)) Top.top
        μ : (i : ι) → Subtype fun x => Membership.mem (iInf fun j => iInf fun x => iIn …
        hμ : Eq (t.sum fun i => ↑(μ i)) 1
        u v : R
        hv : Membership.mem (iInf fun i => iInf fun h => I i) v
        huv : Eq (HAdd.hAdd (HMul.hMul u 1) v) 1
        j : ι
        hj : Membership.mem t j
        hat : Not (Membership.mem t j)
        Hb : ∀ (b : ι), Membership.mem (↑t) b → Ne j b → Eq (Max.max (I j) (I b)) Top. …
        hu : Membership.mem (I j) u
        ⊢ False
      -/
      exact hat hj
      /-
        🎉 no goals
      -/


