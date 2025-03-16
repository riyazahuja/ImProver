lemma tendsto_ite {β : Type*} {p : ι → Prop} [DecidablePred p] {q : Prop} [Decidable q]
    {a b : β} {F G : Filter β}
    (haG : {a}ᶜ ∈ G) (hbF : {b}ᶜ ∈ F) (haF : principal {a} ≤ F) (hbG : principal {b} ≤ G) :
    Tendsto (fun i ↦ if p i then a else b) L (if q then F else G) ↔ ∀ᶠ i in L, p i ↔ q := by
  /-
    ι : Type u_3
    L : Filter ι
    β : Type u_4
    p : ι → Prop
    inst✝¹ : DecidablePred p
    q : Prop
    inst✝ : Decidable q
    a b : β
    F G : Filter β
    haG : Membership.mem G (HasCompl.compl (Singleton.singleton a))
    hbF : Membership.mem F (HasCompl.compl (Singleton.singleton b))
    haF : LE.le (Filter.principal (Singleton.singleton a)) F
    hbG : LE.le (Filter.principal (Singleton.singleton b)) G
    ⊢ Iff (Filter.Tendsto (fun i => ite (p i) a b) L (ite q F G)) (Filter.Eventual …
  -/
  constructor <;> intro h
    /-
      case mp
      ι : Type u_3
      L : Filter ι
      β : Type u_4
      p : ι → Prop
      inst✝¹ : DecidablePred p
      q : Prop
      inst✝ : Decidable q
      a b : β
      F G : Filter β
      haG : Membership.mem G (HasCompl.compl (Singleton.singleton a))
      hbF : Membership.mem F (HasCompl.compl (Singleton.singleton b))
      haF : LE.le (Filter.principal (Singleton.singleton a)) F
      hbG : LE.le (Filter.principal (Singleton.singleton b)) G
      h : Filter.Tendsto (fun i => ite (p i) a b) L (ite q F G)
      ⊢ Filter.Eventually (fun i => Iff (p i) q) L
    -/
  · by_cases hq : q
      /-
        case pos
        ι : Type u_3
        L : Filter ι
        β : Type u_4
        p : ι → Prop
        inst✝¹ : DecidablePred p
        q : Prop
        inst✝ : Decidable q
        a b : β
        F G : Filter β
        haG : Membership.mem G (HasCompl.compl (Singleton.singleton a))
        hbF : Membership.mem F (HasCompl.compl (Singleton.singleton b))
        haF : LE.le (Filter.principal (Singleton.singleton a)) F
        hbG : LE.le (Filter.principal (Singleton.singleton b)) G
        h : Filter.Tendsto (fun i => ite (p i) a b) L (ite q F G)
        hq : q
        ⊢ Filter.Eventually (fun i => Iff (p i) q) L
      -/
    · simp only [hq, ite_true] at h
      /-
        case pos
        ι : Type u_3
        L : Filter ι
        β : Type u_4
        p : ι → Prop
        inst✝¹ : DecidablePred p
        q : Prop
        inst✝ : Decidable q
        a b : β
        F G : Filter β
        haG : Membership.mem G (HasCompl.compl (Singleton.singleton a))
        hbF : Membership.mem F (HasCompl.compl (Singleton.singleton b))
        haF : LE.le (Filter.principal (Singleton.singleton a)) F
        hbG : LE.le (Filter.principal (Singleton.singleton b)) G
        hq : q
        h : Filter.Tendsto (fun i => ite (p i) a b) L F
        ⊢ Filter.Eventually (fun i => Iff (p i) q) L
      -/
      filter_upwards [mem_map.mp (h hbF)] with i hi
      simp only [Set.preimage_compl, Set.mem_compl_iff, Set.mem_preimage, Set.mem_singleton_iff,
        ite_eq_right_iff, not_forall, exists_prop] at hi
      /-
        case h
        ι : Type u_3
        L : Filter ι
        β : Type u_4
        p : ι → Prop
        inst✝¹ : DecidablePred p
        q : Prop
        inst✝ : Decidable q
        a b : β
        F G : Filter β
        haG : Membership.mem G (HasCompl.compl (Singleton.singleton a))
        hbF : Membership.mem F (HasCompl.compl (Singleton.singleton b))
        haF : LE.le (Filter.principal (Singleton.singleton a)) F
        hbG : LE.le (Filter.principal (Singleton.singleton b)) G
        hq : q
        h : Filter.Tendsto (fun i => ite (p i) a b) L F
        i : ι
        hi : And (p i) (Not (Eq a b))
        ⊢ Iff (p i) q
      -/
      tauto
      /-
        🎉 no goals
      -/
      /-
        case neg
        ι : Type u_3
        L : Filter ι
        β : Type u_4
        p : ι → Prop
        inst✝¹ : DecidablePred p
        q : Prop
        inst✝ : Decidable q
        a b : β
        F G : Filter β
        haG : Membership.mem G (HasCompl.compl (Singleton.singleton a))
        hbF : Membership.mem F (HasCompl.compl (Singleton.singleton b))
        haF : LE.le (Filter.principal (Singleton.singleton a)) F
        hbG : LE.le (Filter.principal (Singleton.singleton b)) G
        h : Filter.Tendsto (fun i => ite (p i) a b) L (ite q F G)
        hq : Not q
        ⊢ Filter.Eventually (fun i => Iff (p i) q) L
      -/
    · simp only [hq, ite_false] at h
      /-
        case neg
        ι : Type u_3
        L : Filter ι
        β : Type u_4
        p : ι → Prop
        inst✝¹ : DecidablePred p
        q : Prop
        inst✝ : Decidable q
        a b : β
        F G : Filter β
        haG : Membership.mem G (HasCompl.compl (Singleton.singleton a))
        hbF : Membership.mem F (HasCompl.compl (Singleton.singleton b))
        haF : LE.le (Filter.principal (Singleton.singleton a)) F
        hbG : LE.le (Filter.principal (Singleton.singleton b)) G
        hq : Not q
        h : Filter.Tendsto (fun i => ite (p i) a b) L G
        ⊢ Filter.Eventually (fun i => Iff (p i) q) L
      -/
      filter_upwards [mem_map.mp (h haG)] with i hi
      simp only [Set.preimage_compl, Set.mem_compl_iff, Set.mem_preimage, Set.mem_singleton_iff,
        ite_eq_left_iff, not_forall, exists_prop] at hi
      /-
        case h
        ι : Type u_3
        L : Filter ι
        β : Type u_4
        p : ι → Prop
        inst✝¹ : DecidablePred p
        q : Prop
        inst✝ : Decidable q
        a b : β
        F G : Filter β
        haG : Membership.mem G (HasCompl.compl (Singleton.singleton a))
        hbF : Membership.mem F (HasCompl.compl (Singleton.singleton b))
        haF : LE.le (Filter.principal (Singleton.singleton a)) F
        hbG : LE.le (Filter.principal (Singleton.singleton b)) G
        hq : Not q
        h : Filter.Tendsto (fun i => ite (p i) a b) L G
        i : ι
        hi : And (Not (p i)) (Not (Eq b a))
        ⊢ Iff (p i) q
      -/
      tauto
      /-
        🎉 no goals
      -/
  · have obs : (fun _ ↦ if q then a else b) =ᶠ[L] (fun i ↦ if p i then a else b) := by
      filter_upwards [h] with i hi
      simp only [hi]
    /-
      case mpr
      ι : Type u_3
      L : Filter ι
      β : Type u_4
      p : ι → Prop
      inst✝¹ : DecidablePred p
      q : Prop
      inst✝ : Decidable q
      a b : β
      F G : Filter β
      haG : Membership.mem G (HasCompl.compl (Singleton.singleton a))
      hbF : Membership.mem F (HasCompl.compl (Singleton.singleton b))
      haF : LE.le (Filter.principal (Singleton.singleton a)) F
      hbG : LE.le (Filter.principal (Singleton.singleton b)) G
      h : Filter.Eventually (fun i => Iff (p i) q) L
      obs : L.EventuallyEq (fun x => ite q a b) fun i => ite (p i) a b
      ⊢ Filter.Tendsto (fun i => ite (p i) a b) L (ite q F G)
    -/
    apply Tendsto.congr' obs
    /-
      case mpr
      ι : Type u_3
      L : Filter ι
      β : Type u_4
      p : ι → Prop
      inst✝¹ : DecidablePred p
      q : Prop
      inst✝ : Decidable q
      a b : β
      F G : Filter β
      haG : Membership.mem G (HasCompl.compl (Singleton.singleton a))
      hbF : Membership.mem F (HasCompl.compl (Singleton.singleton b))
      haF : LE.le (Filter.principal (Singleton.singleton a)) F
      hbG : LE.le (Filter.principal (Singleton.singleton b)) G
      h : Filter.Eventually (fun i => Iff (p i) q) L
      obs : L.EventuallyEq (fun x => ite q a b) fun i => ite (p i) a b
      ⊢ Filter.Tendsto (fun x => ite q a b) L (ite q F G)
    -/
    by_cases hq : q
      /-
        case pos
        ι : Type u_3
        L : Filter ι
        β : Type u_4
        p : ι → Prop
        inst✝¹ : DecidablePred p
        q : Prop
        inst✝ : Decidable q
        a b : β
        F G : Filter β
        haG : Membership.mem G (HasCompl.compl (Singleton.singleton a))
        hbF : Membership.mem F (HasCompl.compl (Singleton.singleton b))
        haF : LE.le (Filter.principal (Singleton.singleton a)) F
        hbG : LE.le (Filter.principal (Singleton.singleton b)) G
        h : Filter.Eventually (fun i => Iff (p i) q) L
        obs : L.EventuallyEq (fun x => ite q a b) fun i => ite (p i) a b
        hq : q
        ⊢ Filter.Tendsto (fun x => ite q a b) L (ite q F G)
      -/
    · simp only [hq, iff_true, ite_true]
      /-
        case pos
        ι : Type u_3
        L : Filter ι
        β : Type u_4
        p : ι → Prop
        inst✝¹ : DecidablePred p
        q : Prop
        inst✝ : Decidable q
        a b : β
        F G : Filter β
        haG : Membership.mem G (HasCompl.compl (Singleton.singleton a))
        hbF : Membership.mem F (HasCompl.compl (Singleton.singleton b))
        haF : LE.le (Filter.principal (Singleton.singleton a)) F
        hbG : LE.le (Filter.principal (Singleton.singleton b)) G
        h : Filter.Eventually (fun i => Iff (p i) q) L
        obs : L.EventuallyEq (fun x => ite q a b) fun i => ite (p i) a b
        hq : q
        ⊢ Filter.Tendsto (fun x => a) L F
      -/
      apply le_trans _ haF
      simp only [principal_singleton, le_pure_iff, mem_map, Set.mem_singleton_iff,
        Set.preimage_const_of_mem, univ_mem]
      /-
        case neg
        ι : Type u_3
        L : Filter ι
        β : Type u_4
        p : ι → Prop
        inst✝¹ : DecidablePred p
        q : Prop
        inst✝ : Decidable q
        a b : β
        F G : Filter β
        haG : Membership.mem G (HasCompl.compl (Singleton.singleton a))
        hbF : Membership.mem F (HasCompl.compl (Singleton.singleton b))
        haF : LE.le (Filter.principal (Singleton.singleton a)) F
        hbG : LE.le (Filter.principal (Singleton.singleton b)) G
        h : Filter.Eventually (fun i => Iff (p i) q) L
        obs : L.EventuallyEq (fun x => ite q a b) fun i => ite (p i) a b
        hq : Not q
        ⊢ Filter.Tendsto (fun x => ite q a b) L (ite q F G)
      -/
    · simp only [hq, ite_false]
      /-
        case neg
        ι : Type u_3
        L : Filter ι
        β : Type u_4
        p : ι → Prop
        inst✝¹ : DecidablePred p
        q : Prop
        inst✝ : Decidable q
        a b : β
        F G : Filter β
        haG : Membership.mem G (HasCompl.compl (Singleton.singleton a))
        hbF : Membership.mem F (HasCompl.compl (Singleton.singleton b))
        haF : LE.le (Filter.principal (Singleton.singleton a)) F
        hbG : LE.le (Filter.principal (Singleton.singleton b)) G
        h : Filter.Eventually (fun i => Iff (p i) q) L
        obs : L.EventuallyEq (fun x => ite q a b) fun i => ite (p i) a b
        hq : Not q
        ⊢ Filter.Tendsto (fun x => b) L G
      -/
      apply le_trans _ hbG
      simp only [principal_singleton, le_pure_iff, mem_map, Set.mem_singleton_iff,
        Set.preimage_const_of_mem, univ_mem]


lemma tendsto_indicator_const_apply_iff_eventually' (b : β)
    (nhd_b : {0}ᶜ ∈ 𝓝 b) (nhd_o : {b}ᶜ ∈ 𝓝 0) (x : α) :
    Tendsto (fun i ↦ (As i).indicator (fun (_ : α) ↦ b) x) L (𝓝 (A.indicator (fun (_ : α) ↦ b) x))
      ↔ ∀ᶠ i in L, (x ∈ As i ↔ x ∈ A) := by
  classical
  have heart := @tendsto_ite ι L β (fun i ↦ x ∈ As i) _ (x ∈ A) _ b 0 (𝓝 b) (𝓝 (0 : β))
                nhd_o nhd_b ?_ ?_
  · convert heart
    by_cases hxA : x ∈ A <;> simp [hxA]
  · simp only [principal_singleton, le_def, mem_pure]
    exact fun s s_nhd ↦ mem_of_mem_nhds s_nhd
  · simp only [principal_singleton, le_def, mem_pure]
    exact fun s s_nhd ↦ mem_of_mem_nhds s_nhd


lemma tendsto_indicator_const_iff_forall_eventually'
    (b : β) (nhd_b : {0}ᶜ ∈ 𝓝 b) (nhd_o : {b}ᶜ ∈ 𝓝 0) :
    Tendsto (fun i ↦ (As i).indicator (fun (_ : α) ↦ b)) L (𝓝 (A.indicator (fun (_ : α) ↦ b)))
      ↔ ∀ x, ∀ᶠ i in L, (x ∈ As i ↔ x ∈ A) := by
  /-
    α : Type u_1
    A : Set α
    β : Type u_2
    inst✝¹ : Zero β
    inst✝ : TopologicalSpace β
    ι : Type u_3
    L : Filter ι
    As : ι → Set α
    b : β
    nhd_b : Membership.mem (nhds b) (HasCompl.compl (Singleton.singleton 0))
    nhd_o : Membership.mem (nhds 0) (HasCompl.compl (Singleton.singleton b))
    ⊢ Iff (Filter.Tendsto (fun i => (As i).indicator fun x => b) L (nhds (A.indica …
  -/
  simp_rw [tendsto_pi_nhds]
  /-
    α : Type u_1
    A : Set α
    β : Type u_2
    inst✝¹ : Zero β
    inst✝ : TopologicalSpace β
    ι : Type u_3
    L : Filter ι
    As : ι → Set α
    b : β
    nhd_b : Membership.mem (nhds b) (HasCompl.compl (Singleton.singleton 0))
    nhd_o : Membership.mem (nhds 0) (HasCompl.compl (Singleton.singleton b))
    ⊢ Iff (∀ (x : α), Filter.Tendsto (fun i => (As i).indicator (fun x => b) x) L  …
  -/
  apply forall_congr'
  /-
    case h
    α : Type u_1
    A : Set α
    β : Type u_2
    inst✝¹ : Zero β
    inst✝ : TopologicalSpace β
    ι : Type u_3
    L : Filter ι
    As : ι → Set α
    b : β
    nhd_b : Membership.mem (nhds b) (HasCompl.compl (Singleton.singleton 0))
    nhd_o : Membership.mem (nhds 0) (HasCompl.compl (Singleton.singleton b))
    ⊢ ∀ (a : α), Iff (Filter.Tendsto (fun i => (As i).indicator (fun x => b) a) L  …
  -/
  exact tendsto_indicator_const_apply_iff_eventually' L b nhd_b nhd_o
  /-
    🎉 no goals
  -/


/-- The indicator functions of `Asᵢ` evaluated at `x` tend to the indicator function of `A`
evaluated at `x` if and only if we eventually have the equivalence `x ∈ Asᵢ ↔ x ∈ A`. -/
@[simp] lemma tendsto_indicator_const_apply_iff_eventually [T1Space β] (b : β) [NeZero b]
    (x : α) :
    Tendsto (fun i ↦ (As i).indicator (fun (_ : α) ↦ b) x) L (𝓝 (A.indicator (fun (_ : α) ↦ b) x))
      ↔ ∀ᶠ i in L, (x ∈ As i ↔ x ∈ A) := by
  /-
    α : Type u_1
    A : Set α
    β : Type u_2
    inst✝³ : Zero β
    inst✝² : TopologicalSpace β
    ι : Type u_3
    L : Filter ι
    As : ι → Set α
    inst✝¹ : T1Space β
    b : β
    inst✝ : NeZero b
    x : α
    ⊢ Iff (Filter.Tendsto (fun i => (As i).indicator (fun x => b) x) L (nhds (A.in …
  -/
  apply tendsto_indicator_const_apply_iff_eventually' _ b
    /-
      case nhd_b
      α : Type u_1
      A : Set α
      β : Type u_2
      inst✝³ : Zero β
      inst✝² : TopologicalSpace β
      ι : Type u_3
      L : Filter ι
      As : ι → Set α
      inst✝¹ : T1Space β
      b : β
      inst✝ : NeZero b
      x : α
      ⊢ Membership.mem (nhds b) (HasCompl.compl (Singleton.singleton 0))
    -/
  · simp only [compl_singleton_mem_nhds_iff, ne_eq, NeZero.ne, not_false_eq_true]
    /-
      🎉 no goals
    -/
    /-
      case nhd_o
      α : Type u_1
      A : Set α
      β : Type u_2
      inst✝³ : Zero β
      inst✝² : TopologicalSpace β
      ι : Type u_3
      L : Filter ι
      As : ι → Set α
      inst✝¹ : T1Space β
      b : β
      inst✝ : NeZero b
      x : α
      ⊢ Membership.mem (nhds 0) (HasCompl.compl (Singleton.singleton b))
    -/
  · simp only [compl_singleton_mem_nhds_iff, ne_eq, (NeZero.ne b).symm, not_false_eq_true]
    /-
      🎉 no goals
    -/


/-- The indicator functions of `Asᵢ` tend to the indicator function of `A` pointwise if and only if
for every `x`, we eventually have the equivalence `x ∈ Asᵢ ↔ x ∈ A`. -/
@[simp] lemma tendsto_indicator_const_iff_forall_eventually [T1Space β] (b : β) [NeZero b] :
    Tendsto (fun i ↦ (As i).indicator (fun (_ : α) ↦ b)) L (𝓝 (A.indicator (fun (_ : α) ↦ b)))
      ↔ ∀ x, ∀ᶠ i in L, (x ∈ As i ↔ x ∈ A) := by
  /-
    α : Type u_1
    A : Set α
    β : Type u_2
    inst✝³ : Zero β
    inst✝² : TopologicalSpace β
    ι : Type u_3
    L : Filter ι
    As : ι → Set α
    inst✝¹ : T1Space β
    b : β
    inst✝ : NeZero b
    ⊢ Iff (Filter.Tendsto (fun i => (As i).indicator fun x => b) L (nhds (A.indica …
  -/
  apply tendsto_indicator_const_iff_forall_eventually' _ b
    /-
      case nhd_b
      α : Type u_1
      A : Set α
      β : Type u_2
      inst✝³ : Zero β
      inst✝² : TopologicalSpace β
      ι : Type u_3
      L : Filter ι
      As : ι → Set α
      inst✝¹ : T1Space β
      b : β
      inst✝ : NeZero b
      ⊢ Membership.mem (nhds b) (HasCompl.compl (Singleton.singleton 0))
    -/
  · simp only [compl_singleton_mem_nhds_iff, ne_eq, NeZero.ne, not_false_eq_true]
    /-
      🎉 no goals
    -/
    /-
      case nhd_o
      α : Type u_1
      A : Set α
      β : Type u_2
      inst✝³ : Zero β
      inst✝² : TopologicalSpace β
      ι : Type u_3
      L : Filter ι
      As : ι → Set α
      inst✝¹ : T1Space β
      b : β
      inst✝ : NeZero b
      ⊢ Membership.mem (nhds 0) (HasCompl.compl (Singleton.singleton b))
    -/
  · simp only [compl_singleton_mem_nhds_iff, ne_eq, (NeZero.ne b).symm, not_false_eq_true]
    /-
      🎉 no goals
    -/


lemma tendsto_indicator_const_iff_tendsto_pi_pure'
    (b : β) (nhd_b : {0}ᶜ ∈ 𝓝 b) (nhd_o : {b}ᶜ ∈ 𝓝 0) :
    Tendsto (fun i ↦ (As i).indicator (fun (_ : α) ↦ b)) L (𝓝 (A.indicator (fun (_ : α) ↦ b)))
      ↔ (Tendsto As L <| Filter.pi (pure <| · ∈ A)) := by
  /-
    α : Type u_1
    A : Set α
    β : Type u_2
    inst✝¹ : Zero β
    inst✝ : TopologicalSpace β
    ι : Type u_3
    L : Filter ι
    As : ι → Set α
    b : β
    nhd_b : Membership.mem (nhds b) (HasCompl.compl (Singleton.singleton 0))
    nhd_o : Membership.mem (nhds 0) (HasCompl.compl (Singleton.singleton b))
    ⊢ Iff (Filter.Tendsto (fun i => (As i).indicator fun x => b) L (nhds (A.indica …
  -/
  rw [tendsto_indicator_const_iff_forall_eventually' _ b nhd_b nhd_o, tendsto_pi]
  /-
    α : Type u_1
    A : Set α
    β : Type u_2
    inst✝¹ : Zero β
    inst✝ : TopologicalSpace β
    ι : Type u_3
    L : Filter ι
    As : ι → Set α
    b : β
    nhd_b : Membership.mem (nhds b) (HasCompl.compl (Singleton.singleton 0))
    nhd_o : Membership.mem (nhds 0) (HasCompl.compl (Singleton.singleton b))
    ⊢ Iff (∀ (x : α), Filter.Eventually (fun i => Iff (Membership.mem (As i) x) (M …
  -/
  simp_rw [tendsto_pure]
  /-
    α : Type u_1
    A : Set α
    β : Type u_2
    inst✝¹ : Zero β
    inst✝ : TopologicalSpace β
    ι : Type u_3
    L : Filter ι
    As : ι → Set α
    b : β
    nhd_b : Membership.mem (nhds b) (HasCompl.compl (Singleton.singleton 0))
    nhd_o : Membership.mem (nhds 0) (HasCompl.compl (Singleton.singleton b))
    ⊢ Iff (∀ (x : α), Filter.Eventually (fun i => Iff (Membership.mem (As i) x) (M …
  -/
  aesop
  /-
    🎉 no goals
  -/


lemma tendsto_indicator_const_iff_tendsto_pi_pure [T1Space β] (b : β) [NeZero b] :
    Tendsto (fun i ↦ (As i).indicator (fun (_ : α) ↦ b)) L (𝓝 (A.indicator (fun (_ : α) ↦ b)))
      ↔ (Tendsto As L <| Filter.pi (pure <| · ∈ A)) := by
  /-
    α : Type u_1
    A : Set α
    β : Type u_2
    inst✝³ : Zero β
    inst✝² : TopologicalSpace β
    ι : Type u_3
    L : Filter ι
    As : ι → Set α
    inst✝¹ : T1Space β
    b : β
    inst✝ : NeZero b
    ⊢ Iff (Filter.Tendsto (fun i => (As i).indicator fun x => b) L (nhds (A.indica …
  -/
  rw [tendsto_indicator_const_iff_forall_eventually _ b, tendsto_pi]
  /-
    α : Type u_1
    A : Set α
    β : Type u_2
    inst✝³ : Zero β
    inst✝² : TopologicalSpace β
    ι : Type u_3
    L : Filter ι
    As : ι → Set α
    inst✝¹ : T1Space β
    b : β
    inst✝ : NeZero b
    ⊢ Iff (∀ (x : α), Filter.Eventually (fun i => Iff (Membership.mem (As i) x) (M …
  -/
  simp_rw [tendsto_pure]
  /-
    α : Type u_1
    A : Set α
    β : Type u_2
    inst✝³ : Zero β
    inst✝² : TopologicalSpace β
    ι : Type u_3
    L : Filter ι
    As : ι → Set α
    inst✝¹ : T1Space β
    b : β
    inst✝ : NeZero b
    ⊢ Iff (∀ (x : α), Filter.Eventually (fun i => Iff (Membership.mem (As i) x) (M …
  -/
  aesop
  /-
    🎉 no goals
  -/

