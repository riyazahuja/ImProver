/-- The `ι` indexed basis on `V`, where `ι` is an empty type and `V` is zero-dimensional.

See also `Module.finBasis`.
-/
noncomputable def Basis.ofRankEqZero [Module.Free K V] {ι : Type*} [IsEmpty ι]
    (hV : Module.rank K V = 0) : Basis ι K V :=
  haveI : Subsingleton V := by
    /-
      K : Type u
      V : Type v
      inst✝⁵ : Ring K
      inst✝⁴ : StrongRankCondition K
      inst✝³ : AddCommGroup V
      inst✝² : Module K V
      inst✝¹ : Module.Free K V
      ι : Type u_1
      inst✝ : IsEmpty ι
      hV : Eq (Module.rank K V) 0
      ⊢ Subsingleton V
    -/
    obtain ⟨_, b⟩ := Module.Free.exists_basis (R := K) (M := V)
    /-
      case intro.mk
      K : Type u
      V : Type v
      inst✝⁵ : Ring K
      inst✝⁴ : StrongRankCondition K
      inst✝³ : AddCommGroup V
      inst✝² : Module K V
      inst✝¹ : Module.Free K V
      ι : Type u_1
      inst✝ : IsEmpty ι
      hV : Eq (Module.rank K V) 0
      fst✝ : Type v
      b : Basis fst✝ K V
      ⊢ Subsingleton V
    -/
    haveI := mk_eq_zero_iff.1 (hV ▸ b.mk_eq_rank'')
    /-
      case intro.mk
      K : Type u
      V : Type v
      inst✝⁵ : Ring K
      inst✝⁴ : StrongRankCondition K
      inst✝³ : AddCommGroup V
      inst✝² : Module K V
      inst✝¹ : Module.Free K V
      ι : Type u_1
      inst✝ : IsEmpty ι
      hV : Eq (Module.rank K V) 0
      fst✝ : Type v
      b : Basis fst✝ K V
      this : IsEmpty fst✝
      ⊢ Subsingleton V
    -/
    exact b.repr.toEquiv.subsingleton
    /-
      🎉 no goals
    -/
  Basis.empty _


@[simp]
theorem Basis.ofRankEqZero_apply [Module.Free K V] {ι : Type*} [IsEmpty ι]
    (hV : Module.rank K V = 0) (i : ι) : Basis.ofRankEqZero hV i = 0 := rfl


theorem le_rank_iff_exists_linearIndependent [Module.Free K V] {c : Cardinal} :
    c ≤ Module.rank K V ↔ ∃ s : Set V, #s = c ∧ LinearIndependent K ((↑) : s → V) := by
  /-
    K : Type u
    V : Type v
    inst✝⁴ : Ring K
    inst✝³ : StrongRankCondition K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : Module.Free K V
    c : Cardinal.{v}
    ⊢ Iff (LE.le c (Module.rank K V)) (Exists fun s => And (Eq (Cardinal.mk ↑s) c) …
  -/
  haveI := nontrivial_of_invariantBasisNumber K
  /-
    K : Type u
    V : Type v
    inst✝⁴ : Ring K
    inst✝³ : StrongRankCondition K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : Module.Free K V
    c : Cardinal.{v}
    this : Nontrivial K
    ⊢ Iff (LE.le c (Module.rank K V)) (Exists fun s => And (Eq (Cardinal.mk ↑s) c) …
  -/
  constructor
    /-
      case mp
      K : Type u
      V : Type v
      inst✝⁴ : Ring K
      inst✝³ : StrongRankCondition K
      inst✝² : AddCommGroup V
      inst✝¹ : Module K V
      inst✝ : Module.Free K V
      c : Cardinal.{v}
      this : Nontrivial K
      ⊢ LE.le c (Module.rank K V) → Exists fun s => And (Eq (Cardinal.mk ↑s) c) (Lin …
    -/
  · intro h
    /-
      case mp
      K : Type u
      V : Type v
      inst✝⁴ : Ring K
      inst✝³ : StrongRankCondition K
      inst✝² : AddCommGroup V
      inst✝¹ : Module K V
      inst✝ : Module.Free K V
      c : Cardinal.{v}
      this : Nontrivial K
      h : LE.le c (Module.rank K V)
      ⊢ Exists fun s => And (Eq (Cardinal.mk ↑s) c) (LinearIndependent K Subtype.val)
    -/
    obtain ⟨κ, t'⟩ := Module.Free.exists_basis (R := K) (M := V)
    /-
      case mp.intro.mk
      K : Type u
      V : Type v
      inst✝⁴ : Ring K
      inst✝³ : StrongRankCondition K
      inst✝² : AddCommGroup V
      inst✝¹ : Module K V
      inst✝ : Module.Free K V
      c : Cardinal.{v}
      this : Nontrivial K
      h : LE.le c (Module.rank K V)
      κ : Type v
      t' : Basis κ K V
      ⊢ Exists fun s => And (Eq (Cardinal.mk ↑s) c) (LinearIndependent K Subtype.val)
    -/
    let t := t'.reindexRange
    have : LinearIndependent K ((↑) : Set.range t' → V) := by
      convert t.linearIndependent
      ext; exact (Basis.reindexRange_apply _ _).symm
    /-
      case mp.intro.mk
      K : Type u
      V : Type v
      inst✝⁴ : Ring K
      inst✝³ : StrongRankCondition K
      inst✝² : AddCommGroup V
      inst✝¹ : Module K V
      inst✝ : Module.Free K V
      c : Cardinal.{v}
      this✝ : Nontrivial K
      h : LE.le c (Module.rank K V)
      κ : Type v
      t' : Basis κ K V
      t : Basis (↑(Set.range ⇑t')) K V := t'.reindexRange
      this : LinearIndependent K Subtype.val
      ⊢ Exists fun s => And (Eq (Cardinal.mk ↑s) c) (LinearIndependent K Subtype.val)
    -/
    rw [← t.mk_eq_rank'', le_mk_iff_exists_subset] at h
    /-
      case mp.intro.mk
      K : Type u
      V : Type v
      inst✝⁴ : Ring K
      inst✝³ : StrongRankCondition K
      inst✝² : AddCommGroup V
      inst✝¹ : Module K V
      inst✝ : Module.Free K V
      c : Cardinal.{v}
      this✝ : Nontrivial K
      κ : Type v
      t' : Basis κ K V
      h : Exists fun p => And (HasSubset.Subset p (Set.range ⇑t')) (Eq (Cardinal.mk  …
      t : Basis (↑(Set.range ⇑t')) K V := t'.reindexRange
      this : LinearIndependent K Subtype.val
      ⊢ Exists fun s => And (Eq (Cardinal.mk ↑s) c) (LinearIndependent K Subtype.val)
    -/
    rcases h with ⟨s, hst, hsc⟩
    /-
      case mp.intro.mk.intro.intro
      K : Type u
      V : Type v
      inst✝⁴ : Ring K
      inst✝³ : StrongRankCondition K
      inst✝² : AddCommGroup V
      inst✝¹ : Module K V
      inst✝ : Module.Free K V
      c : Cardinal.{v}
      this✝ : Nontrivial K
      κ : Type v
      t' : Basis κ K V
      t : Basis (↑(Set.range ⇑t')) K V := t'.reindexRange
      this : LinearIndependent K Subtype.val
      s : Set V
      hst : HasSubset.Subset s (Set.range ⇑t')
      hsc : Eq (Cardinal.mk ↑s) c
      ⊢ Exists fun s => And (Eq (Cardinal.mk ↑s) c) (LinearIndependent K Subtype.val)
    -/
    exact ⟨s, hsc, this.mono hst⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      K : Type u
      V : Type v
      inst✝⁴ : Ring K
      inst✝³ : StrongRankCondition K
      inst✝² : AddCommGroup V
      inst✝¹ : Module K V
      inst✝ : Module.Free K V
      c : Cardinal.{v}
      this : Nontrivial K
      ⊢ (Exists fun s => And (Eq (Cardinal.mk ↑s) c) (LinearIndependent K Subtype.va …
    -/
  · rintro ⟨s, rfl, si⟩
    /-
      case mpr.intro.intro
      K : Type u
      V : Type v
      inst✝⁴ : Ring K
      inst✝³ : StrongRankCondition K
      inst✝² : AddCommGroup V
      inst✝¹ : Module K V
      inst✝ : Module.Free K V
      this : Nontrivial K
      s : Set V
      si : LinearIndependent K Subtype.val
      ⊢ LE.le (Cardinal.mk ↑s) (Module.rank K V)
    -/
    exact si.cardinal_le_rank
    /-
      🎉 no goals
    -/


theorem le_rank_iff_exists_linearIndependent_finset
    [Module.Free K V] {n : ℕ} : ↑n ≤ Module.rank K V ↔
    ∃ s : Finset V, s.card = n ∧ LinearIndependent K ((↑) : ↥(s : Set V) → V) := by
  /-
    K : Type u
    V : Type v
    inst✝⁴ : Ring K
    inst✝³ : StrongRankCondition K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : Module.Free K V
    n : Nat
    ⊢ Iff (LE.le (↑n) (Module.rank K V)) (Exists fun s => And (Eq s.card n) (Linea …
  -/
  simp only [le_rank_iff_exists_linearIndependent, mk_set_eq_nat_iff_finset]
  /-
    K : Type u
    V : Type v
    inst✝⁴ : Ring K
    inst✝³ : StrongRankCondition K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : Module.Free K V
    n : Nat
    ⊢ Iff (Exists fun s => And (Exists fun t => And (Eq (↑t) s) (Eq t.card n)) (Li …
  -/
  constructor
    /-
      case mp
      K : Type u
      V : Type v
      inst✝⁴ : Ring K
      inst✝³ : StrongRankCondition K
      inst✝² : AddCommGroup V
      inst✝¹ : Module K V
      inst✝ : Module.Free K V
      n : Nat
      ⊢ (Exists fun s => And (Exists fun t => And (Eq (↑t) s) (Eq t.card n)) (Linear …
    -/
  · rintro ⟨s, ⟨t, rfl, rfl⟩, si⟩
    /-
      case mp.intro.intro.intro.intro
      K : Type u
      V : Type v
      inst✝⁴ : Ring K
      inst✝³ : StrongRankCondition K
      inst✝² : AddCommGroup V
      inst✝¹ : Module K V
      inst✝ : Module.Free K V
      t : Finset V
      si : LinearIndependent K Subtype.val
      ⊢ Exists fun s => And (Eq s.card t.card) (LinearIndependent K Subtype.val)
    -/
    exact ⟨t, rfl, si⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      K : Type u
      V : Type v
      inst✝⁴ : Ring K
      inst✝³ : StrongRankCondition K
      inst✝² : AddCommGroup V
      inst✝¹ : Module K V
      inst✝ : Module.Free K V
      n : Nat
      ⊢ (Exists fun s => And (Eq s.card n) (LinearIndependent K Subtype.val)) → Exis …
    -/
  · rintro ⟨s, rfl, si⟩
    /-
      case mpr.intro.intro
      K : Type u
      V : Type v
      inst✝⁴ : Ring K
      inst✝³ : StrongRankCondition K
      inst✝² : AddCommGroup V
      inst✝¹ : Module K V
      inst✝ : Module.Free K V
      s : Finset V
      si : LinearIndependent K Subtype.val
      ⊢ Exists fun s_1 => And (Exists fun t => And (Eq (↑t) s_1) (Eq t.card s.card)) …
    -/
    exact ⟨s, ⟨s, rfl, rfl⟩, si⟩
    /-
      🎉 no goals
    -/


/-- A vector space has dimension at most `1` if and only if there is a
single vector of which all vectors are multiples. -/
theorem rank_le_one_iff [Module.Free K V] :
    Module.rank K V ≤ 1 ↔ ∃ v₀ : V, ∀ v, ∃ r : K, r • v₀ = v := by
  /-
    K : Type u
    V : Type v
    inst✝⁴ : Ring K
    inst✝³ : StrongRankCondition K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : Module.Free K V
    ⊢ Iff (LE.le (Module.rank K V) 1) (Exists fun v₀ => ∀ (v : V), Exists fun r => …
  -/
  obtain ⟨κ, b⟩ := Module.Free.exists_basis (R := K) (M := V)
  /-
    case intro.mk
    K : Type u
    V : Type v
    inst✝⁴ : Ring K
    inst✝³ : StrongRankCondition K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : Module.Free K V
    κ : Type v
    b : Basis κ K V
    ⊢ Iff (LE.le (Module.rank K V) 1) (Exists fun v₀ => ∀ (v : V), Exists fun r => …
  -/
  constructor
    /-
      case intro.mk.mp
      K : Type u
      V : Type v
      inst✝⁴ : Ring K
      inst✝³ : StrongRankCondition K
      inst✝² : AddCommGroup V
      inst✝¹ : Module K V
      inst✝ : Module.Free K V
      κ : Type v
      b : Basis κ K V
      ⊢ LE.le (Module.rank K V) 1 → Exists fun v₀ => ∀ (v : V), Exists fun r => Eq ( …
    -/
  · intro hd
    /-
      case intro.mk.mp
      K : Type u
      V : Type v
      inst✝⁴ : Ring K
      inst✝³ : StrongRankCondition K
      inst✝² : AddCommGroup V
      inst✝¹ : Module K V
      inst✝ : Module.Free K V
      κ : Type v
      b : Basis κ K V
      hd : LE.le (Module.rank K V) 1
      ⊢ Exists fun v₀ => ∀ (v : V), Exists fun r => Eq (HSMul.hSMul r v₀) v
    -/
    rw [← b.mk_eq_rank'', le_one_iff_subsingleton] at hd
    /-
      case intro.mk.mp
      K : Type u
      V : Type v
      inst✝⁴ : Ring K
      inst✝³ : StrongRankCondition K
      inst✝² : AddCommGroup V
      inst✝¹ : Module K V
      inst✝ : Module.Free K V
      κ : Type v
      b : Basis κ K V
      hd : Subsingleton κ
      ⊢ Exists fun v₀ => ∀ (v : V), Exists fun r => Eq (HSMul.hSMul r v₀) v
    -/
    rcases isEmpty_or_nonempty κ with hb | ⟨⟨i⟩⟩
      /-
        case intro.mk.mp.inl
        K : Type u
        V : Type v
        inst✝⁴ : Ring K
        inst✝³ : StrongRankCondition K
        inst✝² : AddCommGroup V
        inst✝¹ : Module K V
        inst✝ : Module.Free K V
        κ : Type v
        b : Basis κ K V
        hd : Subsingleton κ
        hb : IsEmpty κ
        ⊢ Exists fun v₀ => ∀ (v : V), Exists fun r => Eq (HSMul.hSMul r v₀) v
      -/
    · use 0
      have h' : ∀ v : V, v = 0 := by
        simpa [range_eq_empty, Submodule.eq_bot_iff] using b.span_eq.symm
      /-
        case h
        K : Type u
        V : Type v
        inst✝⁴ : Ring K
        inst✝³ : StrongRankCondition K
        inst✝² : AddCommGroup V
        inst✝¹ : Module K V
        inst✝ : Module.Free K V
        κ : Type v
        b : Basis κ K V
        hd : Subsingleton κ
        hb : IsEmpty κ
        h' : ∀ (v : V), Eq v 0
        ⊢ ∀ (v : V), Exists fun r => Eq (HSMul.hSMul r 0) v
      -/
      intro v
      /-
        case h
        K : Type u
        V : Type v
        inst✝⁴ : Ring K
        inst✝³ : StrongRankCondition K
        inst✝² : AddCommGroup V
        inst✝¹ : Module K V
        inst✝ : Module.Free K V
        κ : Type v
        b : Basis κ K V
        hd : Subsingleton κ
        hb : IsEmpty κ
        h' : ∀ (v : V), Eq v 0
        v : V
        ⊢ Exists fun r => Eq (HSMul.hSMul r 0) v
      -/
      simp [h' v]
      /-
        🎉 no goals
      -/
      /-
        case intro.mk.mp.inr.intro
        K : Type u
        V : Type v
        inst✝⁴ : Ring K
        inst✝³ : StrongRankCondition K
        inst✝² : AddCommGroup V
        inst✝¹ : Module K V
        inst✝ : Module.Free K V
        κ : Type v
        b : Basis κ K V
        hd : Subsingleton κ
        i : κ
        ⊢ Exists fun v₀ => ∀ (v : V), Exists fun r => Eq (HSMul.hSMul r v₀) v
      -/
    · use b i
      have h' : (K ∙ b i) = ⊤ :=
        (subsingleton_range b).eq_singleton_of_mem (mem_range_self i) ▸ b.span_eq
      /-
        case h
        K : Type u
        V : Type v
        inst✝⁴ : Ring K
        inst✝³ : StrongRankCondition K
        inst✝² : AddCommGroup V
        inst✝¹ : Module K V
        inst✝ : Module.Free K V
        κ : Type v
        b : Basis κ K V
        hd : Subsingleton κ
        i : κ
        h' : Eq (Submodule.span K (Singleton.singleton (b i))) Top.top
        ⊢ ∀ (v : V), Exists fun r => Eq (HSMul.hSMul r (b i)) v
      -/
      intro v
      /-
        case h
        K : Type u
        V : Type v
        inst✝⁴ : Ring K
        inst✝³ : StrongRankCondition K
        inst✝² : AddCommGroup V
        inst✝¹ : Module K V
        inst✝ : Module.Free K V
        κ : Type v
        b : Basis κ K V
        hd : Subsingleton κ
        i : κ
        h' : Eq (Submodule.span K (Singleton.singleton (b i))) Top.top
        v : V
        ⊢ Exists fun r => Eq (HSMul.hSMul r (b i)) v
      -/
      have hv : v ∈ (⊤ : Submodule K V) := mem_top
      /-
        case h
        K : Type u
        V : Type v
        inst✝⁴ : Ring K
        inst✝³ : StrongRankCondition K
        inst✝² : AddCommGroup V
        inst✝¹ : Module K V
        inst✝ : Module.Free K V
        κ : Type v
        b : Basis κ K V
        hd : Subsingleton κ
        i : κ
        h' : Eq (Submodule.span K (Singleton.singleton (b i))) Top.top
        v : V
        hv : Membership.mem Top.top v
        ⊢ Exists fun r => Eq (HSMul.hSMul r (b i)) v
      -/
      rwa [← h', mem_span_singleton] at hv
      /-
        🎉 no goals
      -/
    /-
      case intro.mk.mpr
      K : Type u
      V : Type v
      inst✝⁴ : Ring K
      inst✝³ : StrongRankCondition K
      inst✝² : AddCommGroup V
      inst✝¹ : Module K V
      inst✝ : Module.Free K V
      κ : Type v
      b : Basis κ K V
      ⊢ (Exists fun v₀ => ∀ (v : V), Exists fun r => Eq (HSMul.hSMul r v₀) v) → LE.l …
    -/
  · rintro ⟨v₀, hv₀⟩
    have h : (K ∙ v₀) = ⊤ := by
      ext
      simp [mem_span_singleton, hv₀]
    /-
      case intro.mk.mpr.intro
      K : Type u
      V : Type v
      inst✝⁴ : Ring K
      inst✝³ : StrongRankCondition K
      inst✝² : AddCommGroup V
      inst✝¹ : Module K V
      inst✝ : Module.Free K V
      κ : Type v
      b : Basis κ K V
      v₀ : V
      hv₀ : ∀ (v : V), Exists fun r => Eq (HSMul.hSMul r v₀) v
      h : Eq (Submodule.span K (Singleton.singleton v₀)) Top.top
      ⊢ LE.le (Module.rank K V) 1
    -/
    rw [← rank_top, ← h]
    /-
      case intro.mk.mpr.intro
      K : Type u
      V : Type v
      inst✝⁴ : Ring K
      inst✝³ : StrongRankCondition K
      inst✝² : AddCommGroup V
      inst✝¹ : Module K V
      inst✝ : Module.Free K V
      κ : Type v
      b : Basis κ K V
      v₀ : V
      hv₀ : ∀ (v : V), Exists fun r => Eq (HSMul.hSMul r v₀) v
      h : Eq (Submodule.span K (Singleton.singleton v₀)) Top.top
      ⊢ LE.le (Module.rank K (Subtype fun x => Membership.mem (Submodule.span K (Sin …
    -/
    refine (rank_span_le _).trans_eq ?_
    /-
      case intro.mk.mpr.intro
      K : Type u
      V : Type v
      inst✝⁴ : Ring K
      inst✝³ : StrongRankCondition K
      inst✝² : AddCommGroup V
      inst✝¹ : Module K V
      inst✝ : Module.Free K V
      κ : Type v
      b : Basis κ K V
      v₀ : V
      hv₀ : ∀ (v : V), Exists fun r => Eq (HSMul.hSMul r v₀) v
      h : Eq (Submodule.span K (Singleton.singleton v₀)) Top.top
      ⊢ Eq (Cardinal.mk ↑(Singleton.singleton v₀)) 1
    -/
    simp
    /-
      🎉 no goals
    -/


/-- A vector space has dimension `1` if and only if there is a
single non-zero vector of which all vectors are multiples. -/
theorem rank_eq_one_iff [Module.Free K V] :
    Module.rank K V = 1 ↔ ∃ v₀ : V, v₀ ≠ 0 ∧ ∀ v, ∃ r : K, r • v₀ = v := by
  /-
    K : Type u
    V : Type v
    inst✝⁴ : Ring K
    inst✝³ : StrongRankCondition K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : Module.Free K V
    ⊢ Iff (Eq (Module.rank K V) 1) (Exists fun v₀ => And (Ne v₀ 0) (∀ (v : V), Exi …
  -/
  haveI := nontrivial_of_invariantBasisNumber K
  /-
    K : Type u
    V : Type v
    inst✝⁴ : Ring K
    inst✝³ : StrongRankCondition K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : Module.Free K V
    this : Nontrivial K
    ⊢ Iff (Eq (Module.rank K V) 1) (Exists fun v₀ => And (Ne v₀ 0) (∀ (v : V), Exi …
  -/
  refine ⟨fun h ↦ ?_, fun ⟨v₀, h, hv⟩ ↦ (rank_le_one_iff.2 ⟨v₀, hv⟩).antisymm ?_⟩
    /-
      case refine_1
      K : Type u
      V : Type v
      inst✝⁴ : Ring K
      inst✝³ : StrongRankCondition K
      inst✝² : AddCommGroup V
      inst✝¹ : Module K V
      inst✝ : Module.Free K V
      this : Nontrivial K
      h : Eq (Module.rank K V) 1
      ⊢ Exists fun v₀ => And (Ne v₀ 0) (∀ (v : V), Exists fun r => Eq (HSMul.hSMul r …
    -/
  · obtain ⟨v₀, hv⟩ := rank_le_one_iff.1 h.le
    /-
      case refine_1.intro
      K : Type u
      V : Type v
      inst✝⁴ : Ring K
      inst✝³ : StrongRankCondition K
      inst✝² : AddCommGroup V
      inst✝¹ : Module K V
      inst✝ : Module.Free K V
      this : Nontrivial K
      h : Eq (Module.rank K V) 1
      v₀ : V
      hv : ∀ (v : V), Exists fun r => Eq (HSMul.hSMul r v₀) v
      ⊢ Exists fun v₀ => And (Ne v₀ 0) (∀ (v : V), Exists fun r => Eq (HSMul.hSMul r …
    -/
    refine ⟨v₀, fun hzero ↦ ?_, hv⟩
    /-
      case refine_1.intro
      K : Type u
      V : Type v
      inst✝⁴ : Ring K
      inst✝³ : StrongRankCondition K
      inst✝² : AddCommGroup V
      inst✝¹ : Module K V
      inst✝ : Module.Free K V
      this : Nontrivial K
      h : Eq (Module.rank K V) 1
      v₀ : V
      hv : ∀ (v : V), Exists fun r => Eq (HSMul.hSMul r v₀) v
      hzero : Eq v₀ 0
      ⊢ False
    -/
    simp_rw [hzero, smul_zero, exists_const] at hv
    /-
      case refine_1.intro
      K : Type u
      V : Type v
      inst✝⁴ : Ring K
      inst✝³ : StrongRankCondition K
      inst✝² : AddCommGroup V
      inst✝¹ : Module K V
      inst✝ : Module.Free K V
      this : Nontrivial K
      h : Eq (Module.rank K V) 1
      v₀ : V
      hzero : Eq v₀ 0
      hv : ∀ (v : V), Eq 0 v
      ⊢ False
    -/
    haveI : Subsingleton V := .intro fun _ _ ↦ by simp_rw [← hv]
    /-
      case refine_1.intro
      K : Type u
      V : Type v
      inst✝⁴ : Ring K
      inst✝³ : StrongRankCondition K
      inst✝² : AddCommGroup V
      inst✝¹ : Module K V
      inst✝ : Module.Free K V
      this✝ : Nontrivial K
      h : Eq (Module.rank K V) 1
      v₀ : V
      hzero : Eq v₀ 0
      hv : ∀ (v : V), Eq 0 v
      this : Subsingleton V
      ⊢ False
    -/
    exact one_ne_zero (h ▸ rank_subsingleton' K V)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      K : Type u
      V : Type v
      inst✝⁴ : Ring K
      inst✝³ : StrongRankCondition K
      inst✝² : AddCommGroup V
      inst✝¹ : Module K V
      inst✝ : Module.Free K V
      this : Nontrivial K
      x✝ : Exists fun v₀ => And (Ne v₀ 0) (∀ (v : V), Exists fun r => Eq (HSMul.hSMu …
      v₀ : V
      h : Ne v₀ 0
      hv : ∀ (v : V), Exists fun r => Eq (HSMul.hSMul r v₀) v
      ⊢ LE.le 1 (Module.rank K V)
    -/
  · by_contra H
    /-
      case refine_2
      K : Type u
      V : Type v
      inst✝⁴ : Ring K
      inst✝³ : StrongRankCondition K
      inst✝² : AddCommGroup V
      inst✝¹ : Module K V
      inst✝ : Module.Free K V
      this : Nontrivial K
      x✝ : Exists fun v₀ => And (Ne v₀ 0) (∀ (v : V), Exists fun r => Eq (HSMul.hSMu …
      v₀ : V
      h : Ne v₀ 0
      hv : ∀ (v : V), Exists fun r => Eq (HSMul.hSMul r v₀) v
      H : Not (LE.le 1 (Module.rank K V))
      ⊢ False
    -/
    rw [not_le, lt_one_iff_zero] at H
    /-
      case refine_2
      K : Type u
      V : Type v
      inst✝⁴ : Ring K
      inst✝³ : StrongRankCondition K
      inst✝² : AddCommGroup V
      inst✝¹ : Module K V
      inst✝ : Module.Free K V
      this : Nontrivial K
      x✝ : Exists fun v₀ => And (Ne v₀ 0) (∀ (v : V), Exists fun r => Eq (HSMul.hSMu …
      v₀ : V
      h : Ne v₀ 0
      hv : ∀ (v : V), Exists fun r => Eq (HSMul.hSMul r v₀) v
      H : Eq (Module.rank K V) 0
      ⊢ False
    -/
    obtain ⟨κ, b⟩ := Module.Free.exists_basis (R := K) (M := V)
    /-
      case refine_2.intro.mk
      K : Type u
      V : Type v
      inst✝⁴ : Ring K
      inst✝³ : StrongRankCondition K
      inst✝² : AddCommGroup V
      inst✝¹ : Module K V
      inst✝ : Module.Free K V
      this : Nontrivial K
      x✝ : Exists fun v₀ => And (Ne v₀ 0) (∀ (v : V), Exists fun r => Eq (HSMul.hSMu …
      v₀ : V
      h : Ne v₀ 0
      hv : ∀ (v : V), Exists fun r => Eq (HSMul.hSMul r v₀) v
      H : Eq (Module.rank K V) 0
      κ : Type v
      b : Basis κ K V
      ⊢ False
    -/
    haveI := mk_eq_zero_iff.1 (H ▸ b.mk_eq_rank'')
    /-
      case refine_2.intro.mk
      K : Type u
      V : Type v
      inst✝⁴ : Ring K
      inst✝³ : StrongRankCondition K
      inst✝² : AddCommGroup V
      inst✝¹ : Module K V
      inst✝ : Module.Free K V
      this✝ : Nontrivial K
      x✝ : Exists fun v₀ => And (Ne v₀ 0) (∀ (v : V), Exists fun r => Eq (HSMul.hSMu …
      v₀ : V
      h : Ne v₀ 0
      hv : ∀ (v : V), Exists fun r => Eq (HSMul.hSMul r v₀) v
      H : Eq (Module.rank K V) 0
      κ : Type v
      b : Basis κ K V
      this : IsEmpty κ
      ⊢ False
    -/
    haveI := b.repr.toEquiv.subsingleton
    /-
      case refine_2.intro.mk
      K : Type u
      V : Type v
      inst✝⁴ : Ring K
      inst✝³ : StrongRankCondition K
      inst✝² : AddCommGroup V
      inst✝¹ : Module K V
      inst✝ : Module.Free K V
      this✝¹ : Nontrivial K
      x✝ : Exists fun v₀ => And (Ne v₀ 0) (∀ (v : V), Exists fun r => Eq (HSMul.hSMu …
      v₀ : V
      h : Ne v₀ 0
      hv : ∀ (v : V), Exists fun r => Eq (HSMul.hSMul r v₀) v
      H : Eq (Module.rank K V) 0
      κ : Type v
      b : Basis κ K V
      this✝ : IsEmpty κ
      this : Subsingleton V
      ⊢ False
    -/
    exact h (Subsingleton.elim _ _)
    /-
      🎉 no goals
    -/


/-- A submodule has dimension at most `1` if and only if there is a
single vector in the submodule such that the submodule is contained in
its span. -/
theorem rank_submodule_le_one_iff (s : Submodule K V) [Module.Free K s] :
    Module.rank K s ≤ 1 ↔ ∃ v₀ ∈ s, s ≤ K ∙ v₀ := by
  /-
    K : Type u
    V : Type v
    inst✝⁴ : Ring K
    inst✝³ : StrongRankCondition K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    s : Submodule K V
    inst✝ : Module.Free K (Subtype fun x => Membership.mem s x)
    ⊢ Iff (LE.le (Module.rank K (Subtype fun x => Membership.mem s x)) 1) (Exists  …
  -/
  simp_rw [rank_le_one_iff, le_span_singleton_iff]
  /-
    K : Type u
    V : Type v
    inst✝⁴ : Ring K
    inst✝³ : StrongRankCondition K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    s : Submodule K V
    inst✝ : Module.Free K (Subtype fun x => Membership.mem s x)
    ⊢ Iff (Exists fun v₀ => ∀ (v : Subtype fun x => Membership.mem s x), Exists fu …
  -/
  constructor
    /-
      case mp
      K : Type u
      V : Type v
      inst✝⁴ : Ring K
      inst✝³ : StrongRankCondition K
      inst✝² : AddCommGroup V
      inst✝¹ : Module K V
      s : Submodule K V
      inst✝ : Module.Free K (Subtype fun x => Membership.mem s x)
      ⊢ (Exists fun v₀ => ∀ (v : Subtype fun x => Membership.mem s x), Exists fun r  …
    -/
  · rintro ⟨⟨v₀, hv₀⟩, h⟩
    /-
      case mp.intro.mk
      K : Type u
      V : Type v
      inst✝⁴ : Ring K
      inst✝³ : StrongRankCondition K
      inst✝² : AddCommGroup V
      inst✝¹ : Module K V
      s : Submodule K V
      inst✝ : Module.Free K (Subtype fun x => Membership.mem s x)
      v₀ : V
      hv₀ : Membership.mem s v₀
      h : ∀ (v : Subtype fun x => Membership.mem s x), Exists fun r => Eq (HSMul.hSM …
      ⊢ Exists fun v₀ => And (Membership.mem s v₀) (∀ (v : V), Membership.mem s v →  …
    -/
    use v₀, hv₀
    /-
      case right
      K : Type u
      V : Type v
      inst✝⁴ : Ring K
      inst✝³ : StrongRankCondition K
      inst✝² : AddCommGroup V
      inst✝¹ : Module K V
      s : Submodule K V
      inst✝ : Module.Free K (Subtype fun x => Membership.mem s x)
      v₀ : V
      hv₀ : Membership.mem s v₀
      h : ∀ (v : Subtype fun x => Membership.mem s x), Exists fun r => Eq (HSMul.hSM …
      ⊢ ∀ (v : V), Membership.mem s v → Exists fun r => Eq (HSMul.hSMul r v₀) v
    -/
    intro v hv
    /-
      case right
      K : Type u
      V : Type v
      inst✝⁴ : Ring K
      inst✝³ : StrongRankCondition K
      inst✝² : AddCommGroup V
      inst✝¹ : Module K V
      s : Submodule K V
      inst✝ : Module.Free K (Subtype fun x => Membership.mem s x)
      v₀ : V
      hv₀ : Membership.mem s v₀
      h : ∀ (v : Subtype fun x => Membership.mem s x), Exists fun r => Eq (HSMul.hSM …
      v : V
      hv : Membership.mem s v
      ⊢ Exists fun r => Eq (HSMul.hSMul r v₀) v
    -/
    obtain ⟨r, hr⟩ := h ⟨v, hv⟩
    /-
      case right.intro
      K : Type u
      V : Type v
      inst✝⁴ : Ring K
      inst✝³ : StrongRankCondition K
      inst✝² : AddCommGroup V
      inst✝¹ : Module K V
      s : Submodule K V
      inst✝ : Module.Free K (Subtype fun x => Membership.mem s x)
      v₀ : V
      hv₀ : Membership.mem s v₀
      h : ∀ (v : Subtype fun x => Membership.mem s x), Exists fun r => Eq (HSMul.hSM …
      v : V
      hv : Membership.mem s v
      r : K
      hr : Eq (HSMul.hSMul r ⟨v₀, hv₀⟩) ⟨v, hv⟩
      ⊢ Exists fun r => Eq (HSMul.hSMul r v₀) v
    -/
    use r
    /-
      case h
      K : Type u
      V : Type v
      inst✝⁴ : Ring K
      inst✝³ : StrongRankCondition K
      inst✝² : AddCommGroup V
      inst✝¹ : Module K V
      s : Submodule K V
      inst✝ : Module.Free K (Subtype fun x => Membership.mem s x)
      v₀ : V
      hv₀ : Membership.mem s v₀
      h : ∀ (v : Subtype fun x => Membership.mem s x), Exists fun r => Eq (HSMul.hSM …
      v : V
      hv : Membership.mem s v
      r : K
      hr : Eq (HSMul.hSMul r ⟨v₀, hv₀⟩) ⟨v, hv⟩
      ⊢ Eq (HSMul.hSMul r v₀) v
    -/
    rwa [Subtype.ext_iff, coe_smul] at hr
    /-
      🎉 no goals
    -/
    /-
      case mpr
      K : Type u
      V : Type v
      inst✝⁴ : Ring K
      inst✝³ : StrongRankCondition K
      inst✝² : AddCommGroup V
      inst✝¹ : Module K V
      s : Submodule K V
      inst✝ : Module.Free K (Subtype fun x => Membership.mem s x)
      ⊢ (Exists fun v₀ => And (Membership.mem s v₀) (∀ (v : V), Membership.mem s v → …
    -/
  · rintro ⟨v₀, hv₀, h⟩
    /-
      case mpr.intro.intro
      K : Type u
      V : Type v
      inst✝⁴ : Ring K
      inst✝³ : StrongRankCondition K
      inst✝² : AddCommGroup V
      inst✝¹ : Module K V
      s : Submodule K V
      inst✝ : Module.Free K (Subtype fun x => Membership.mem s x)
      v₀ : V
      hv₀ : Membership.mem s v₀
      h : ∀ (v : V), Membership.mem s v → Exists fun r => Eq (HSMul.hSMul r v₀) v
      ⊢ Exists fun v₀ => ∀ (v : Subtype fun x => Membership.mem s x), Exists fun r = …
    -/
    use ⟨v₀, hv₀⟩
    /-
      case h
      K : Type u
      V : Type v
      inst✝⁴ : Ring K
      inst✝³ : StrongRankCondition K
      inst✝² : AddCommGroup V
      inst✝¹ : Module K V
      s : Submodule K V
      inst✝ : Module.Free K (Subtype fun x => Membership.mem s x)
      v₀ : V
      hv₀ : Membership.mem s v₀
      h : ∀ (v : V), Membership.mem s v → Exists fun r => Eq (HSMul.hSMul r v₀) v
      ⊢ ∀ (v : Subtype fun x => Membership.mem s x), Exists fun r => Eq (HSMul.hSMul …
    -/
    rintro ⟨v, hv⟩
    /-
      case h.mk
      K : Type u
      V : Type v
      inst✝⁴ : Ring K
      inst✝³ : StrongRankCondition K
      inst✝² : AddCommGroup V
      inst✝¹ : Module K V
      s : Submodule K V
      inst✝ : Module.Free K (Subtype fun x => Membership.mem s x)
      v₀ : V
      hv₀ : Membership.mem s v₀
      h : ∀ (v : V), Membership.mem s v → Exists fun r => Eq (HSMul.hSMul r v₀) v
      v : V
      hv : Membership.mem s v
      ⊢ Exists fun r => Eq (HSMul.hSMul r ⟨v₀, hv₀⟩) ⟨v, hv⟩
    -/
    obtain ⟨r, hr⟩ := h v hv
    /-
      case h.mk.intro
      K : Type u
      V : Type v
      inst✝⁴ : Ring K
      inst✝³ : StrongRankCondition K
      inst✝² : AddCommGroup V
      inst✝¹ : Module K V
      s : Submodule K V
      inst✝ : Module.Free K (Subtype fun x => Membership.mem s x)
      v₀ : V
      hv₀ : Membership.mem s v₀
      h : ∀ (v : V), Membership.mem s v → Exists fun r => Eq (HSMul.hSMul r v₀) v
      v : V
      hv : Membership.mem s v
      r : K
      hr : Eq (HSMul.hSMul r v₀) v
      ⊢ Exists fun r => Eq (HSMul.hSMul r ⟨v₀, hv₀⟩) ⟨v, hv⟩
    -/
    use r
    /-
      case h
      K : Type u
      V : Type v
      inst✝⁴ : Ring K
      inst✝³ : StrongRankCondition K
      inst✝² : AddCommGroup V
      inst✝¹ : Module K V
      s : Submodule K V
      inst✝ : Module.Free K (Subtype fun x => Membership.mem s x)
      v₀ : V
      hv₀ : Membership.mem s v₀
      h : ∀ (v : V), Membership.mem s v → Exists fun r => Eq (HSMul.hSMul r v₀) v
      v : V
      hv : Membership.mem s v
      r : K
      hr : Eq (HSMul.hSMul r v₀) v
      ⊢ Eq (HSMul.hSMul r ⟨v₀, hv₀⟩) ⟨v, hv⟩
    -/
    rwa [Subtype.ext_iff, coe_smul]
    /-
      🎉 no goals
    -/


/-- A submodule has dimension `1` if and only if there is a
single non-zero vector in the submodule such that the submodule is contained in
its span. -/
theorem rank_submodule_eq_one_iff (s : Submodule K V) [Module.Free K s] :
    Module.rank K s = 1 ↔ ∃ v₀ ∈ s, v₀ ≠ 0 ∧ s ≤ K ∙ v₀ := by
  /-
    K : Type u
    V : Type v
    inst✝⁴ : Ring K
    inst✝³ : StrongRankCondition K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    s : Submodule K V
    inst✝ : Module.Free K (Subtype fun x => Membership.mem s x)
    ⊢ Iff (Eq (Module.rank K (Subtype fun x => Membership.mem s x)) 1) (Exists fun …
  -/
  simp_rw [rank_eq_one_iff, le_span_singleton_iff]
  refine ⟨fun ⟨⟨v₀, hv₀⟩, H, h⟩ ↦ ⟨v₀, hv₀, fun h' ↦ by
    simp only [h', ne_eq] at H; exact H rfl, fun v hv ↦ ?_⟩,
    fun ⟨v₀, hv₀, H, h⟩ ↦ ⟨⟨v₀, hv₀⟩,
      fun h' ↦ H (by rwa [AddSubmonoid.mk_eq_zero] at h'), fun ⟨v, hv⟩ ↦ ?_⟩⟩
    /-
      case refine_1
      K : Type u
      V : Type v
      inst✝⁴ : Ring K
      inst✝³ : StrongRankCondition K
      inst✝² : AddCommGroup V
      inst✝¹ : Module K V
      s : Submodule K V
      inst✝ : Module.Free K (Subtype fun x => Membership.mem s x)
      x✝ : Exists fun v₀ => And (Ne v₀ 0) (∀ (v : Subtype fun x => Membership.mem s  …
      v₀ : V
      hv₀ : Membership.mem s v₀
      H : Ne ⟨v₀, hv₀⟩ 0
      h : ∀ (v : Subtype fun x => Membership.mem s x), Exists fun r => Eq (HSMul.hSM …
      v : V
      hv : Membership.mem s v
      ⊢ Exists fun r => Eq (HSMul.hSMul r v₀) v
    -/
  · obtain ⟨r, hr⟩ := h ⟨v, hv⟩
    /-
      case refine_1.intro
      K : Type u
      V : Type v
      inst✝⁴ : Ring K
      inst✝³ : StrongRankCondition K
      inst✝² : AddCommGroup V
      inst✝¹ : Module K V
      s : Submodule K V
      inst✝ : Module.Free K (Subtype fun x => Membership.mem s x)
      x✝ : Exists fun v₀ => And (Ne v₀ 0) (∀ (v : Subtype fun x => Membership.mem s  …
      v₀ : V
      hv₀ : Membership.mem s v₀
      H : Ne ⟨v₀, hv₀⟩ 0
      h : ∀ (v : Subtype fun x => Membership.mem s x), Exists fun r => Eq (HSMul.hSM …
      v : V
      hv : Membership.mem s v
      r : K
      hr : Eq (HSMul.hSMul r ⟨v₀, hv₀⟩) ⟨v, hv⟩
      ⊢ Exists fun r => Eq (HSMul.hSMul r v₀) v
    -/
    exact ⟨r, by rwa [Subtype.ext_iff, coe_smul] at hr⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      K : Type u
      V : Type v
      inst✝⁴ : Ring K
      inst✝³ : StrongRankCondition K
      inst✝² : AddCommGroup V
      inst✝¹ : Module K V
      s : Submodule K V
      inst✝ : Module.Free K (Subtype fun x => Membership.mem s x)
      x✝¹ : Exists fun v₀ => And (Membership.mem s v₀) (And (Ne v₀ 0) (∀ (v : V), Me …
      v₀ : V
      hv₀ : Membership.mem s v₀
      H : Ne v₀ 0
      h : ∀ (v : V), Membership.mem s v → Exists fun r => Eq (HSMul.hSMul r v₀) v
      x✝ : Subtype fun x => Membership.mem s x
      v : V
      hv : Membership.mem s v
      ⊢ Exists fun r => Eq (HSMul.hSMul r ⟨v₀, hv₀⟩) ⟨v, hv⟩
    -/
  · obtain ⟨r, hr⟩ := h v hv
    /-
      case refine_2.intro
      K : Type u
      V : Type v
      inst✝⁴ : Ring K
      inst✝³ : StrongRankCondition K
      inst✝² : AddCommGroup V
      inst✝¹ : Module K V
      s : Submodule K V
      inst✝ : Module.Free K (Subtype fun x => Membership.mem s x)
      x✝¹ : Exists fun v₀ => And (Membership.mem s v₀) (And (Ne v₀ 0) (∀ (v : V), Me …
      v₀ : V
      hv₀ : Membership.mem s v₀
      H : Ne v₀ 0
      h : ∀ (v : V), Membership.mem s v → Exists fun r => Eq (HSMul.hSMul r v₀) v
      x✝ : Subtype fun x => Membership.mem s x
      v : V
      hv : Membership.mem s v
      r : K
      hr : Eq (HSMul.hSMul r v₀) v
      ⊢ Exists fun r => Eq (HSMul.hSMul r ⟨v₀, hv₀⟩) ⟨v, hv⟩
    -/
    exact ⟨r, by rwa [Subtype.ext_iff, coe_smul]⟩
    /-
      🎉 no goals
    -/


/-- A submodule has dimension at most `1` if and only if there is a
single vector, not necessarily in the submodule, such that the
submodule is contained in its span. -/
theorem rank_submodule_le_one_iff' (s : Submodule K V) [Module.Free K s] :
    Module.rank K s ≤ 1 ↔ ∃ v₀, s ≤ K ∙ v₀ := by
  /-
    K : Type u
    V : Type v
    inst✝⁴ : Ring K
    inst✝³ : StrongRankCondition K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    s : Submodule K V
    inst✝ : Module.Free K (Subtype fun x => Membership.mem s x)
    ⊢ Iff (LE.le (Module.rank K (Subtype fun x => Membership.mem s x)) 1) (Exists  …
  -/
  haveI := nontrivial_of_invariantBasisNumber K
  /-
    K : Type u
    V : Type v
    inst✝⁴ : Ring K
    inst✝³ : StrongRankCondition K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    s : Submodule K V
    inst✝ : Module.Free K (Subtype fun x => Membership.mem s x)
    this : Nontrivial K
    ⊢ Iff (LE.le (Module.rank K (Subtype fun x => Membership.mem s x)) 1) (Exists  …
  -/
  constructor
    /-
      case mp
      K : Type u
      V : Type v
      inst✝⁴ : Ring K
      inst✝³ : StrongRankCondition K
      inst✝² : AddCommGroup V
      inst✝¹ : Module K V
      s : Submodule K V
      inst✝ : Module.Free K (Subtype fun x => Membership.mem s x)
      this : Nontrivial K
      ⊢ LE.le (Module.rank K (Subtype fun x => Membership.mem s x)) 1 → Exists fun v …
    -/
  · rw [rank_submodule_le_one_iff]
    /-
      case mp
      K : Type u
      V : Type v
      inst✝⁴ : Ring K
      inst✝³ : StrongRankCondition K
      inst✝² : AddCommGroup V
      inst✝¹ : Module K V
      s : Submodule K V
      inst✝ : Module.Free K (Subtype fun x => Membership.mem s x)
      this : Nontrivial K
      ⊢ (Exists fun v₀ => And (Membership.mem s v₀) (LE.le s (Submodule.span K (Sing …
    -/
    rintro ⟨v₀, _, h⟩
    /-
      case mp.intro.intro
      K : Type u
      V : Type v
      inst✝⁴ : Ring K
      inst✝³ : StrongRankCondition K
      inst✝² : AddCommGroup V
      inst✝¹ : Module K V
      s : Submodule K V
      inst✝ : Module.Free K (Subtype fun x => Membership.mem s x)
      this : Nontrivial K
      v₀ : V
      left✝ : Membership.mem s v₀
      h : LE.le s (Submodule.span K (Singleton.singleton v₀))
      ⊢ Exists fun v₀ => LE.le s (Submodule.span K (Singleton.singleton v₀))
    -/
    exact ⟨v₀, h⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      K : Type u
      V : Type v
      inst✝⁴ : Ring K
      inst✝³ : StrongRankCondition K
      inst✝² : AddCommGroup V
      inst✝¹ : Module K V
      s : Submodule K V
      inst✝ : Module.Free K (Subtype fun x => Membership.mem s x)
      this : Nontrivial K
      ⊢ (Exists fun v₀ => LE.le s (Submodule.span K (Singleton.singleton v₀))) → LE. …
    -/
  · rintro ⟨v₀, h⟩
    /-
      case mpr.intro
      K : Type u
      V : Type v
      inst✝⁴ : Ring K
      inst✝³ : StrongRankCondition K
      inst✝² : AddCommGroup V
      inst✝¹ : Module K V
      s : Submodule K V
      inst✝ : Module.Free K (Subtype fun x => Membership.mem s x)
      this : Nontrivial K
      v₀ : V
      h : LE.le s (Submodule.span K (Singleton.singleton v₀))
      ⊢ LE.le (Module.rank K (Subtype fun x => Membership.mem s x)) 1
    -/
    obtain ⟨κ, b⟩ := Module.Free.exists_basis (R := K) (M := s)
    simpa [b.mk_eq_rank''] using b.linearIndependent.map' _ (ker_inclusion _ _ h)
      |>.cardinal_le_rank.trans (rank_span_le {v₀})


theorem Submodule.rank_le_one_iff_isPrincipal (W : Submodule K V) [Module.Free K W] :
    Module.rank K W ≤ 1 ↔ W.IsPrincipal := by
  simp only [rank_le_one_iff, Submodule.isPrincipal_iff, le_antisymm_iff, le_span_singleton_iff,
    span_singleton_le_iff_mem]
  /-
    K : Type u
    V : Type v
    inst✝⁴ : Ring K
    inst✝³ : StrongRankCondition K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    W : Submodule K V
    inst✝ : Module.Free K (Subtype fun x => Membership.mem W x)
    ⊢ Iff (Exists fun v₀ => ∀ (v : Subtype fun x => Membership.mem W x), Exists fu …
  -/
  constructor
    /-
      case mp
      K : Type u
      V : Type v
      inst✝⁴ : Ring K
      inst✝³ : StrongRankCondition K
      inst✝² : AddCommGroup V
      inst✝¹ : Module K V
      W : Submodule K V
      inst✝ : Module.Free K (Subtype fun x => Membership.mem W x)
      ⊢ (Exists fun v₀ => ∀ (v : Subtype fun x => Membership.mem W x), Exists fun r  …
    -/
  · rintro ⟨⟨m, hm⟩, hm'⟩
    /-
      case mp.intro.mk
      K : Type u
      V : Type v
      inst✝⁴ : Ring K
      inst✝³ : StrongRankCondition K
      inst✝² : AddCommGroup V
      inst✝¹ : Module K V
      W : Submodule K V
      inst✝ : Module.Free K (Subtype fun x => Membership.mem W x)
      m : V
      hm : Membership.mem W m
      hm' : ∀ (v : Subtype fun x => Membership.mem W x), Exists fun r => Eq (HSMul.h …
      ⊢ Exists fun a => And (∀ (v : V), Membership.mem W v → Exists fun r => Eq (HSM …
    -/
    choose f hf using hm'
    /-
      case mp.intro.mk
      K : Type u
      V : Type v
      inst✝⁴ : Ring K
      inst✝³ : StrongRankCondition K
      inst✝² : AddCommGroup V
      inst✝¹ : Module K V
      W : Submodule K V
      inst✝ : Module.Free K (Subtype fun x => Membership.mem W x)
      m : V
      hm : Membership.mem W m
      f : (Subtype fun x => Membership.mem W x) → K
      hf : ∀ (v : Subtype fun x => Membership.mem W x), Eq (HSMul.hSMul (f v) ⟨m, hm …
      ⊢ Exists fun a => And (∀ (v : V), Membership.mem W v → Exists fun r => Eq (HSM …
    -/
    exact ⟨m, ⟨fun v hv => ⟨f ⟨v, hv⟩, congr_arg ((↑) : W → V) (hf ⟨v, hv⟩)⟩, hm⟩⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      K : Type u
      V : Type v
      inst✝⁴ : Ring K
      inst✝³ : StrongRankCondition K
      inst✝² : AddCommGroup V
      inst✝¹ : Module K V
      W : Submodule K V
      inst✝ : Module.Free K (Subtype fun x => Membership.mem W x)
      ⊢ (Exists fun a => And (∀ (v : V), Membership.mem W v → Exists fun r => Eq (HS …
    -/
  · rintro ⟨a, ⟨h, ha⟩⟩
    /-
      case mpr.intro.intro
      K : Type u
      V : Type v
      inst✝⁴ : Ring K
      inst✝³ : StrongRankCondition K
      inst✝² : AddCommGroup V
      inst✝¹ : Module K V
      W : Submodule K V
      inst✝ : Module.Free K (Subtype fun x => Membership.mem W x)
      a : V
      h : ∀ (v : V), Membership.mem W v → Exists fun r => Eq (HSMul.hSMul r a) v
      ha : Membership.mem W a
      ⊢ Exists fun v₀ => ∀ (v : Subtype fun x => Membership.mem W x), Exists fun r = …
    -/
    choose f hf using h
    /-
      case mpr.intro.intro
      K : Type u
      V : Type v
      inst✝⁴ : Ring K
      inst✝³ : StrongRankCondition K
      inst✝² : AddCommGroup V
      inst✝¹ : Module K V
      W : Submodule K V
      inst✝ : Module.Free K (Subtype fun x => Membership.mem W x)
      a : V
      ha : Membership.mem W a
      f : (v : V) → Membership.mem W v → K
      hf : ∀ (v : V) (a_1 : Membership.mem W v), Eq (HSMul.hSMul (f v a_1) a) v
      ⊢ Exists fun v₀ => ∀ (v : Subtype fun x => Membership.mem W x), Exists fun r = …
    -/
    exact ⟨⟨a, ha⟩, fun v => ⟨f v.1 v.2, Subtype.ext (hf v.1 v.2)⟩⟩
    /-
      🎉 no goals
    -/


theorem Module.rank_le_one_iff_top_isPrincipal [Module.Free K V] :
    Module.rank K V ≤ 1 ↔ (⊤ : Submodule K V).IsPrincipal := by
  /-
    K : Type u
    V : Type v
    inst✝⁴ : Ring K
    inst✝³ : StrongRankCondition K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : Module.Free K V
    ⊢ Iff (LE.le (Module.rank K V) 1) Top.top.IsPrincipal
  -/
  haveI := Module.Free.of_equiv (topEquiv (R := K) (M := V)).symm
  /-
    K : Type u
    V : Type v
    inst✝⁴ : Ring K
    inst✝³ : StrongRankCondition K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : Module.Free K V
    this : Module.Free K (Subtype fun x => Membership.mem Top.top x)
    ⊢ Iff (LE.le (Module.rank K V) 1) Top.top.IsPrincipal
  -/
  rw [← Submodule.rank_le_one_iff_isPrincipal, rank_top]
  /-
    🎉 no goals
  -/


/-- A module has dimension 1 iff there is some `v : V` so `{v}` is a basis.
-/
theorem finrank_eq_one_iff [Module.Free K V] (ι : Type*) [Unique ι] :
    finrank K V = 1 ↔ Nonempty (Basis ι K V) := by
  /-
    K : Type u
    V : Type v
    inst✝⁵ : Ring K
    inst✝⁴ : StrongRankCondition K
    inst✝³ : AddCommGroup V
    inst✝² : Module K V
    inst✝¹ : Module.Free K V
    ι : Type u_1
    inst✝ : Unique ι
    ⊢ Iff (Eq (Module.finrank K V) 1) (Nonempty (Basis ι K V))
  -/
  constructor
    /-
      case mp
      K : Type u
      V : Type v
      inst✝⁵ : Ring K
      inst✝⁴ : StrongRankCondition K
      inst✝³ : AddCommGroup V
      inst✝² : Module K V
      inst✝¹ : Module.Free K V
      ι : Type u_1
      inst✝ : Unique ι
      ⊢ Eq (Module.finrank K V) 1 → Nonempty (Basis ι K V)
    -/
  · intro h
    /-
      case mp
      K : Type u
      V : Type v
      inst✝⁵ : Ring K
      inst✝⁴ : StrongRankCondition K
      inst✝³ : AddCommGroup V
      inst✝² : Module K V
      inst✝¹ : Module.Free K V
      ι : Type u_1
      inst✝ : Unique ι
      h : Eq (Module.finrank K V) 1
      ⊢ Nonempty (Basis ι K V)
    -/
    exact ⟨Module.basisUnique ι h⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      K : Type u
      V : Type v
      inst✝⁵ : Ring K
      inst✝⁴ : StrongRankCondition K
      inst✝³ : AddCommGroup V
      inst✝² : Module K V
      inst✝¹ : Module.Free K V
      ι : Type u_1
      inst✝ : Unique ι
      ⊢ Nonempty (Basis ι K V) → Eq (Module.finrank K V) 1
    -/
  · rintro ⟨b⟩
    /-
      case mpr.intro
      K : Type u
      V : Type v
      inst✝⁵ : Ring K
      inst✝⁴ : StrongRankCondition K
      inst✝³ : AddCommGroup V
      inst✝² : Module K V
      inst✝¹ : Module.Free K V
      ι : Type u_1
      inst✝ : Unique ι
      b : Basis ι K V
      ⊢ Eq (Module.finrank K V) 1
    -/
    simpa using finrank_eq_card_basis b
    /-
      🎉 no goals
    -/


/-- A module has dimension 1 iff there is some nonzero `v : V` so every vector is a multiple of `v`.
-/
theorem finrank_eq_one_iff' [Module.Free K V] :
    finrank K V = 1 ↔ ∃ v ≠ 0, ∀ w : V, ∃ c : K, c • v = w := by
  /-
    K : Type u
    V : Type v
    inst✝⁴ : Ring K
    inst✝³ : StrongRankCondition K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : Module.Free K V
    ⊢ Iff (Eq (Module.finrank K V) 1) (Exists fun v => And (Ne v 0) (∀ (w : V), Ex …
  -/
  rw [← rank_eq_one_iff]
  /-
    K : Type u
    V : Type v
    inst✝⁴ : Ring K
    inst✝³ : StrongRankCondition K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : Module.Free K V
    ⊢ Iff (Eq (Module.finrank K V) 1) (Eq (Module.rank K V) 1)
  -/
  exact toNat_eq_iff one_ne_zero
  /-
    🎉 no goals
  -/


/-- A finite dimensional module has dimension at most 1 iff
there is some `v : V` so every vector is a multiple of `v`.
-/
theorem finrank_le_one_iff [Module.Free K V] [Module.Finite K V] :
    finrank K V ≤ 1 ↔ ∃ v : V, ∀ w : V, ∃ c : K, c • v = w := by
  /-
    K : Type u
    V : Type v
    inst✝⁵ : Ring K
    inst✝⁴ : StrongRankCondition K
    inst✝³ : AddCommGroup V
    inst✝² : Module K V
    inst✝¹ : Module.Free K V
    inst✝ : Module.Finite K V
    ⊢ Iff (LE.le (Module.finrank K V) 1) (Exists fun v => ∀ (w : V), Exists fun c  …
  -/
  rw [← rank_le_one_iff, ← finrank_eq_rank, Nat.cast_le_one]
  /-
    🎉 no goals
  -/


theorem Submodule.finrank_le_one_iff_isPrincipal
    (W : Submodule K V) [Module.Free K W] [Module.Finite K W] :
    finrank K W ≤ 1 ↔ W.IsPrincipal := by
  /-
    K : Type u
    V : Type v
    inst✝⁵ : Ring K
    inst✝⁴ : StrongRankCondition K
    inst✝³ : AddCommGroup V
    inst✝² : Module K V
    W : Submodule K V
    inst✝¹ : Module.Free K (Subtype fun x => Membership.mem W x)
    inst✝ : Module.Finite K (Subtype fun x => Membership.mem W x)
    ⊢ Iff (LE.le (Module.finrank K (Subtype fun x => Membership.mem W x)) 1) W.IsP …
  -/
  rw [← W.rank_le_one_iff_isPrincipal, ← finrank_eq_rank, Nat.cast_le_one]
  /-
    🎉 no goals
  -/


theorem Module.finrank_le_one_iff_top_isPrincipal [Module.Free K V] [Module.Finite K V] :
    finrank K V ≤ 1 ↔ (⊤ : Submodule K V).IsPrincipal := by
  /-
    K : Type u
    V : Type v
    inst✝⁵ : Ring K
    inst✝⁴ : StrongRankCondition K
    inst✝³ : AddCommGroup V
    inst✝² : Module K V
    inst✝¹ : Module.Free K V
    inst✝ : Module.Finite K V
    ⊢ Iff (LE.le (Module.finrank K V) 1) Top.top.IsPrincipal
  -/
  rw [← Module.rank_le_one_iff_top_isPrincipal, ← finrank_eq_rank, Nat.cast_le_one]
  /-
    🎉 no goals
  -/


variable (K V) in
theorem lift_cardinalMk_eq_lift_cardinalMk_field_pow_lift_rank [Module.Free K V]
    [Module.Finite K V] : lift.{u} #V = lift.{v} #K ^ lift.{u} (Module.rank K V) := by
  /-
    K : Type u
    V : Type v
    inst✝⁵ : Ring K
    inst✝⁴ : StrongRankCondition K
    inst✝³ : AddCommGroup V
    inst✝² : Module K V
    inst✝¹ : Module.Free K V
    inst✝ : Module.Finite K V
    ⊢ Eq (Cardinal.lift.{u, v} (Cardinal.mk V)) (HPow.hPow (Cardinal.lift.{v, u} ( …
  -/
  haveI := nontrivial_of_invariantBasisNumber K
  /-
    K : Type u
    V : Type v
    inst✝⁵ : Ring K
    inst✝⁴ : StrongRankCondition K
    inst✝³ : AddCommGroup V
    inst✝² : Module K V
    inst✝¹ : Module.Free K V
    inst✝ : Module.Finite K V
    this : Nontrivial K
    ⊢ Eq (Cardinal.lift.{u, v} (Cardinal.mk V)) (HPow.hPow (Cardinal.lift.{v, u} ( …
  -/
  obtain ⟨s, hs⟩ := Module.Free.exists_basis (R := K) (M := V)
  -- `Module.Finite.finite_basis` is in a much later file, so we copy its proof to here
  haveI : Finite s := by
    obtain ⟨t, ht⟩ := ‹Module.Finite K V›
    exact basis_finite_of_finite_spans _ t.finite_toSet ht hs
  /-
    case intro.mk
    K : Type u
    V : Type v
    inst✝⁵ : Ring K
    inst✝⁴ : StrongRankCondition K
    inst✝³ : AddCommGroup V
    inst✝² : Module K V
    inst✝¹ : Module.Free K V
    inst✝ : Module.Finite K V
    this✝ : Nontrivial K
    s : Type v
    hs : Basis s K V
    this : Finite s
    ⊢ Eq (Cardinal.lift.{u, v} (Cardinal.mk V)) (HPow.hPow (Cardinal.lift.{v, u} ( …
  -/
  have := lift_mk_eq'.2 ⟨hs.repr.toEquiv⟩
  rwa [Finsupp.equivFunOnFinite.cardinal_eq, mk_arrow, hs.mk_eq_rank'', lift_power, lift_lift,
    lift_lift, lift_umax] at this


@[deprecated (since := "2024-11-10")]
alias lift_cardinal_mk_eq_lift_cardinal_mk_field_pow_lift_rank :=
  lift_cardinalMk_eq_lift_cardinalMk_field_pow_lift_rank


theorem cardinalMk_eq_cardinalMk_field_pow_rank (K V : Type u) [Ring K] [StrongRankCondition K]
    [AddCommGroup V] [Module K V] [Module.Free K V] [Module.Finite K V] :
    #V = #K ^ Module.rank K V := by
  /-
    K V : Type u
    inst✝⁵ : Ring K
    inst✝⁴ : StrongRankCondition K
    inst✝³ : AddCommGroup V
    inst✝² : Module K V
    inst✝¹ : Module.Free K V
    inst✝ : Module.Finite K V
    ⊢ Eq (Cardinal.mk V) (HPow.hPow (Cardinal.mk K) (Module.rank K V))
  -/
  simpa using lift_cardinalMk_eq_lift_cardinalMk_field_pow_lift_rank K V
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-10")]
alias cardinal_mk_eq_cardinal_mk_field_pow_rank := cardinalMk_eq_cardinalMk_field_pow_rank


variable (K V) in
theorem cardinal_lt_aleph0_of_finiteDimensional [Finite K] [Module.Free K V] [Module.Finite K V] :
    #V < ℵ₀ := by
  /-
    K : Type u
    V : Type v
    inst✝⁶ : Ring K
    inst✝⁵ : StrongRankCondition K
    inst✝⁴ : AddCommGroup V
    inst✝³ : Module K V
    inst✝² : Finite K
    inst✝¹ : Module.Free K V
    inst✝ : Module.Finite K V
    ⊢ LT.lt (Cardinal.mk V) Cardinal.aleph0
  -/
  rw [← lift_lt_aleph0.{v, u}, lift_cardinalMk_eq_lift_cardinalMk_field_pow_lift_rank K V]
  exact power_lt_aleph0 (lift_lt_aleph0.2 (lt_aleph0_of_finite K))
    (lift_lt_aleph0.2 (rank_lt_aleph0 K V))


theorem eq_bot_of_rank_le_one (h : Module.rank F S ≤ 1) [Module.Free F S] : S = ⊥ := by
  /-
    F : Type u_1
    E : Type u_2
    inst✝⁴ : CommRing F
    inst✝³ : StrongRankCondition F
    inst✝² : Ring E
    inst✝¹ : Algebra F E
    S : Subalgebra F E
    h : LE.le (Module.rank F (Subtype fun x => Membership.mem S x)) 1
    inst✝ : Module.Free F (Subtype fun x => Membership.mem S x)
    ⊢ Eq S Bot.bot
  -/
  nontriviality E
  /-
    F : Type u_1
    E : Type u_2
    inst✝⁴ : CommRing F
    inst✝³ : StrongRankCondition F
    inst✝² : Ring E
    inst✝¹ : Algebra F E
    S : Subalgebra F E
    h : LE.le (Module.rank F (Subtype fun x => Membership.mem S x)) 1
    inst✝ : Module.Free F (Subtype fun x => Membership.mem S x)
    a✝ : Nontrivial E
    ⊢ Eq S Bot.bot
  -/
  obtain ⟨κ, b⟩ := Module.Free.exists_basis (R := F) (M := S)
  /-
    case intro.mk
    F : Type u_1
    E : Type u_2
    inst✝⁴ : CommRing F
    inst✝³ : StrongRankCondition F
    inst✝² : Ring E
    inst✝¹ : Algebra F E
    S : Subalgebra F E
    h : LE.le (Module.rank F (Subtype fun x => Membership.mem S x)) 1
    inst✝ : Module.Free F (Subtype fun x => Membership.mem S x)
    a✝ : Nontrivial E
    κ : Type u_2
    b : Basis κ F (Subtype fun x => Membership.mem S x)
    ⊢ Eq S Bot.bot
  -/
  by_cases h1 : Module.rank F S = 1
    /-
      case pos
      F : Type u_1
      E : Type u_2
      inst✝⁴ : CommRing F
      inst✝³ : StrongRankCondition F
      inst✝² : Ring E
      inst✝¹ : Algebra F E
      S : Subalgebra F E
      h : LE.le (Module.rank F (Subtype fun x => Membership.mem S x)) 1
      inst✝ : Module.Free F (Subtype fun x => Membership.mem S x)
      a✝ : Nontrivial E
      κ : Type u_2
      b : Basis κ F (Subtype fun x => Membership.mem S x)
      h1 : Eq (Module.rank F (Subtype fun x => Membership.mem S x)) 1
      ⊢ Eq S Bot.bot
    -/
  · refine bot_unique fun x hx ↦ Algebra.mem_bot.2 ?_
    /-
      case pos
      F : Type u_1
      E : Type u_2
      inst✝⁴ : CommRing F
      inst✝³ : StrongRankCondition F
      inst✝² : Ring E
      inst✝¹ : Algebra F E
      S : Subalgebra F E
      h : LE.le (Module.rank F (Subtype fun x => Membership.mem S x)) 1
      inst✝ : Module.Free F (Subtype fun x => Membership.mem S x)
      a✝ : Nontrivial E
      κ : Type u_2
      b : Basis κ F (Subtype fun x => Membership.mem S x)
      h1 : Eq (Module.rank F (Subtype fun x => Membership.mem S x)) 1
      x : E
      hx : Membership.mem S x
      ⊢ Membership.mem (Set.range ⇑(algebraMap F E)) x
    -/
    rw [← b.mk_eq_rank'', eq_one_iff_unique, ← unique_iff_subsingleton_and_nonempty] at h1
    /-
      case pos
      F : Type u_1
      E : Type u_2
      inst✝⁴ : CommRing F
      inst✝³ : StrongRankCondition F
      inst✝² : Ring E
      inst✝¹ : Algebra F E
      S : Subalgebra F E
      h : LE.le (Module.rank F (Subtype fun x => Membership.mem S x)) 1
      inst✝ : Module.Free F (Subtype fun x => Membership.mem S x)
      a✝ : Nontrivial E
      κ : Type u_2
      b : Basis κ F (Subtype fun x => Membership.mem S x)
      h1 : Nonempty (Unique κ)
      x : E
      hx : Membership.mem S x
      ⊢ Membership.mem (Set.range ⇑(algebraMap F E)) x
    -/
    obtain ⟨h1⟩ := h1
    obtain ⟨y, hy⟩ := (bijective_algebraMap_of_linearEquiv (b.repr ≪≫ₗ
      Finsupp.LinearEquiv.finsuppUnique _ _ _).symm).surjective ⟨x, hx⟩
    /-
      case pos.intro.intro
      F : Type u_1
      E : Type u_2
      inst✝⁴ : CommRing F
      inst✝³ : StrongRankCondition F
      inst✝² : Ring E
      inst✝¹ : Algebra F E
      S : Subalgebra F E
      h : LE.le (Module.rank F (Subtype fun x => Membership.mem S x)) 1
      inst✝ : Module.Free F (Subtype fun x => Membership.mem S x)
      a✝ : Nontrivial E
      κ : Type u_2
      b : Basis κ F (Subtype fun x => Membership.mem S x)
      x : E
      hx : Membership.mem S x
      h1 : Unique κ
      y : F
      hy : Eq ((algebraMap F (Subtype fun x => Membership.mem S x)) y) ⟨x, hx⟩
      ⊢ Membership.mem (Set.range ⇑(algebraMap F E)) x
    -/
    exact ⟨y, congr(Subtype.val $(hy))⟩
    /-
      🎉 no goals
    -/
  /-
    case neg
    F : Type u_1
    E : Type u_2
    inst✝⁴ : CommRing F
    inst✝³ : StrongRankCondition F
    inst✝² : Ring E
    inst✝¹ : Algebra F E
    S : Subalgebra F E
    h : LE.le (Module.rank F (Subtype fun x => Membership.mem S x)) 1
    inst✝ : Module.Free F (Subtype fun x => Membership.mem S x)
    a✝ : Nontrivial E
    κ : Type u_2
    b : Basis κ F (Subtype fun x => Membership.mem S x)
    h1 : Not (Eq (Module.rank F (Subtype fun x => Membership.mem S x)) 1)
    ⊢ Eq S Bot.bot
  -/
  haveI := mk_eq_zero_iff.1 (b.mk_eq_rank''.symm ▸ lt_one_iff_zero.1 (h.lt_of_ne h1))
  /-
    case neg
    F : Type u_1
    E : Type u_2
    inst✝⁴ : CommRing F
    inst✝³ : StrongRankCondition F
    inst✝² : Ring E
    inst✝¹ : Algebra F E
    S : Subalgebra F E
    h : LE.le (Module.rank F (Subtype fun x => Membership.mem S x)) 1
    inst✝ : Module.Free F (Subtype fun x => Membership.mem S x)
    a✝ : Nontrivial E
    κ : Type u_2
    b : Basis κ F (Subtype fun x => Membership.mem S x)
    h1 : Not (Eq (Module.rank F (Subtype fun x => Membership.mem S x)) 1)
    this : IsEmpty κ
    ⊢ Eq S Bot.bot
  -/
  haveI := b.repr.toEquiv.subsingleton
  /-
    case neg
    F : Type u_1
    E : Type u_2
    inst✝⁴ : CommRing F
    inst✝³ : StrongRankCondition F
    inst✝² : Ring E
    inst✝¹ : Algebra F E
    S : Subalgebra F E
    h : LE.le (Module.rank F (Subtype fun x => Membership.mem S x)) 1
    inst✝ : Module.Free F (Subtype fun x => Membership.mem S x)
    a✝ : Nontrivial E
    κ : Type u_2
    b : Basis κ F (Subtype fun x => Membership.mem S x)
    h1 : Not (Eq (Module.rank F (Subtype fun x => Membership.mem S x)) 1)
    this✝ : IsEmpty κ
    this : Subsingleton (Subtype fun x => Membership.mem S x)
    ⊢ Eq S Bot.bot
  -/
  exact False.elim <| one_ne_zero congr(S.val $(Subsingleton.elim 1 0))
  /-
    🎉 no goals
  -/


theorem eq_bot_of_finrank_one (h : finrank F S = 1) [Module.Free F S] : S = ⊥ := by
  /-
    F : Type u_1
    E : Type u_2
    inst✝⁴ : CommRing F
    inst✝³ : StrongRankCondition F
    inst✝² : Ring E
    inst✝¹ : Algebra F E
    S : Subalgebra F E
    h : Eq (Module.finrank F (Subtype fun x => Membership.mem S x)) 1
    inst✝ : Module.Free F (Subtype fun x => Membership.mem S x)
    ⊢ Eq S Bot.bot
  -/
  refine Subalgebra.eq_bot_of_rank_le_one ?_
  /-
    F : Type u_1
    E : Type u_2
    inst✝⁴ : CommRing F
    inst✝³ : StrongRankCondition F
    inst✝² : Ring E
    inst✝¹ : Algebra F E
    S : Subalgebra F E
    h : Eq (Module.finrank F (Subtype fun x => Membership.mem S x)) 1
    inst✝ : Module.Free F (Subtype fun x => Membership.mem S x)
    ⊢ LE.le (Module.rank F (Subtype fun x => Membership.mem S x)) 1
  -/
  rw [finrank, toNat_eq_one] at h
  /-
    F : Type u_1
    E : Type u_2
    inst✝⁴ : CommRing F
    inst✝³ : StrongRankCondition F
    inst✝² : Ring E
    inst✝¹ : Algebra F E
    S : Subalgebra F E
    h : Eq (Module.rank F (Subtype fun x => Membership.mem S x)) 1
    inst✝ : Module.Free F (Subtype fun x => Membership.mem S x)
    ⊢ LE.le (Module.rank F (Subtype fun x => Membership.mem S x)) 1
  -/
  rw [h]
  /-
    🎉 no goals
  -/


@[simp]
theorem rank_eq_one_iff [Nontrivial E] [Module.Free F S] : Module.rank F S = 1 ↔ S = ⊥ := by
  /-
    F : Type u_1
    E : Type u_2
    inst✝⁵ : CommRing F
    inst✝⁴ : StrongRankCondition F
    inst✝³ : Ring E
    inst✝² : Algebra F E
    S : Subalgebra F E
    inst✝¹ : Nontrivial E
    inst✝ : Module.Free F (Subtype fun x => Membership.mem S x)
    ⊢ Iff (Eq (Module.rank F (Subtype fun x => Membership.mem S x)) 1) (Eq S Bot.b …
  -/
  refine ⟨fun h ↦ Subalgebra.eq_bot_of_rank_le_one h.le, ?_⟩
  /-
    F : Type u_1
    E : Type u_2
    inst✝⁵ : CommRing F
    inst✝⁴ : StrongRankCondition F
    inst✝³ : Ring E
    inst✝² : Algebra F E
    S : Subalgebra F E
    inst✝¹ : Nontrivial E
    inst✝ : Module.Free F (Subtype fun x => Membership.mem S x)
    ⊢ Eq S Bot.bot → Eq (Module.rank F (Subtype fun x => Membership.mem S x)) 1
  -/
  rintro rfl
  /-
    F : Type u_1
    E : Type u_2
    inst✝⁵ : CommRing F
    inst✝⁴ : StrongRankCondition F
    inst✝³ : Ring E
    inst✝² : Algebra F E
    inst✝¹ : Nontrivial E
    inst✝ : Module.Free F (Subtype fun x => Membership.mem Bot.bot x)
    ⊢ Eq (Module.rank F (Subtype fun x => Membership.mem Bot.bot x)) 1
  -/
  obtain ⟨κ, b⟩ := Module.Free.exists_basis (R := F) (M := (⊥ : Subalgebra F E))
  /-
    case intro.mk
    F : Type u_1
    E : Type u_2
    inst✝⁵ : CommRing F
    inst✝⁴ : StrongRankCondition F
    inst✝³ : Ring E
    inst✝² : Algebra F E
    inst✝¹ : Nontrivial E
    inst✝ : Module.Free F (Subtype fun x => Membership.mem Bot.bot x)
    κ : Type u_2
    b : Basis κ F (Subtype fun x => Membership.mem Bot.bot x)
    ⊢ Eq (Module.rank F (Subtype fun x => Membership.mem Bot.bot x)) 1
  -/
  refine le_antisymm ?_ ?_
    /-
      case intro.mk.refine_1
      F : Type u_1
      E : Type u_2
      inst✝⁵ : CommRing F
      inst✝⁴ : StrongRankCondition F
      inst✝³ : Ring E
      inst✝² : Algebra F E
      inst✝¹ : Nontrivial E
      inst✝ : Module.Free F (Subtype fun x => Membership.mem Bot.bot x)
      κ : Type u_2
      b : Basis κ F (Subtype fun x => Membership.mem Bot.bot x)
      ⊢ LE.le (Module.rank F (Subtype fun x => Membership.mem Bot.bot x)) 1
    -/
  · have := lift_rank_range_le (Algebra.linearMap F E)
    rwa [← one_eq_range, rank_self, lift_one, lift_le_one_iff,
      ← Algebra.toSubmodule_bot, rank_toSubmodule] at this
    /-
      case intro.mk.refine_2
      F : Type u_1
      E : Type u_2
      inst✝⁵ : CommRing F
      inst✝⁴ : StrongRankCondition F
      inst✝³ : Ring E
      inst✝² : Algebra F E
      inst✝¹ : Nontrivial E
      inst✝ : Module.Free F (Subtype fun x => Membership.mem Bot.bot x)
      κ : Type u_2
      b : Basis κ F (Subtype fun x => Membership.mem Bot.bot x)
      ⊢ LE.le 1 (Module.rank F (Subtype fun x => Membership.mem Bot.bot x))
    -/
  · by_contra H
    /-
      case intro.mk.refine_2
      F : Type u_1
      E : Type u_2
      inst✝⁵ : CommRing F
      inst✝⁴ : StrongRankCondition F
      inst✝³ : Ring E
      inst✝² : Algebra F E
      inst✝¹ : Nontrivial E
      inst✝ : Module.Free F (Subtype fun x => Membership.mem Bot.bot x)
      κ : Type u_2
      b : Basis κ F (Subtype fun x => Membership.mem Bot.bot x)
      H : Not (LE.le 1 (Module.rank F (Subtype fun x => Membership.mem Bot.bot x)))
      ⊢ False
    -/
    rw [not_le, lt_one_iff_zero] at H
    /-
      case intro.mk.refine_2
      F : Type u_1
      E : Type u_2
      inst✝⁵ : CommRing F
      inst✝⁴ : StrongRankCondition F
      inst✝³ : Ring E
      inst✝² : Algebra F E
      inst✝¹ : Nontrivial E
      inst✝ : Module.Free F (Subtype fun x => Membership.mem Bot.bot x)
      κ : Type u_2
      b : Basis κ F (Subtype fun x => Membership.mem Bot.bot x)
      H : Eq (Module.rank F (Subtype fun x => Membership.mem Bot.bot x)) 0
      ⊢ False
    -/
    haveI := mk_eq_zero_iff.1 (H ▸ b.mk_eq_rank'')
    /-
      case intro.mk.refine_2
      F : Type u_1
      E : Type u_2
      inst✝⁵ : CommRing F
      inst✝⁴ : StrongRankCondition F
      inst✝³ : Ring E
      inst✝² : Algebra F E
      inst✝¹ : Nontrivial E
      inst✝ : Module.Free F (Subtype fun x => Membership.mem Bot.bot x)
      κ : Type u_2
      b : Basis κ F (Subtype fun x => Membership.mem Bot.bot x)
      H : Eq (Module.rank F (Subtype fun x => Membership.mem Bot.bot x)) 0
      this : IsEmpty κ
      ⊢ False
    -/
    haveI := b.repr.toEquiv.subsingleton
    /-
      case intro.mk.refine_2
      F : Type u_1
      E : Type u_2
      inst✝⁵ : CommRing F
      inst✝⁴ : StrongRankCondition F
      inst✝³ : Ring E
      inst✝² : Algebra F E
      inst✝¹ : Nontrivial E
      inst✝ : Module.Free F (Subtype fun x => Membership.mem Bot.bot x)
      κ : Type u_2
      b : Basis κ F (Subtype fun x => Membership.mem Bot.bot x)
      H : Eq (Module.rank F (Subtype fun x => Membership.mem Bot.bot x)) 0
      this✝ : IsEmpty κ
      this : Subsingleton (Subtype fun x => Membership.mem Bot.bot x)
      ⊢ False
    -/
    exact one_ne_zero congr((⊥ : Subalgebra F E).val $(Subsingleton.elim 1 0))
    /-
      🎉 no goals
    -/


@[simp]
theorem finrank_eq_one_iff [Nontrivial E] [Module.Free F S] : finrank F S = 1 ↔ S = ⊥ := by
  /-
    F : Type u_1
    E : Type u_2
    inst✝⁵ : CommRing F
    inst✝⁴ : StrongRankCondition F
    inst✝³ : Ring E
    inst✝² : Algebra F E
    S : Subalgebra F E
    inst✝¹ : Nontrivial E
    inst✝ : Module.Free F (Subtype fun x => Membership.mem S x)
    ⊢ Iff (Eq (Module.finrank F (Subtype fun x => Membership.mem S x)) 1) (Eq S Bo …
  -/
  rw [← Subalgebra.rank_eq_one_iff]
  /-
    F : Type u_1
    E : Type u_2
    inst✝⁵ : CommRing F
    inst✝⁴ : StrongRankCondition F
    inst✝³ : Ring E
    inst✝² : Algebra F E
    S : Subalgebra F E
    inst✝¹ : Nontrivial E
    inst✝ : Module.Free F (Subtype fun x => Membership.mem S x)
    ⊢ Iff (Eq (Module.finrank F (Subtype fun x => Membership.mem S x)) 1) (Eq (Mod …
  -/
  exact toNat_eq_iff one_ne_zero
  /-
    🎉 no goals
  -/


theorem bot_eq_top_iff_rank_eq_one [Nontrivial E] [Module.Free F E] :
    (⊥ : Subalgebra F E) = ⊤ ↔ Module.rank F E = 1 := by
  /-
    F : Type u_1
    E : Type u_2
    inst✝⁵ : CommRing F
    inst✝⁴ : StrongRankCondition F
    inst✝³ : Ring E
    inst✝² : Algebra F E
    inst✝¹ : Nontrivial E
    inst✝ : Module.Free F E
    ⊢ Iff (Eq Bot.bot Top.top) (Eq (Module.rank F E) 1)
  -/
  haveI := Module.Free.of_equiv (Subalgebra.topEquiv (R := F) (A := E)).toLinearEquiv.symm
  -- Porting note: removed `subalgebra_top_rank_eq_submodule_top_rank`
  /-
    F : Type u_1
    E : Type u_2
    inst✝⁵ : CommRing F
    inst✝⁴ : StrongRankCondition F
    inst✝³ : Ring E
    inst✝² : Algebra F E
    inst✝¹ : Nontrivial E
    inst✝ : Module.Free F E
    this : Module.Free F (Subtype fun x => Membership.mem Top.top x)
    ⊢ Iff (Eq Bot.bot Top.top) (Eq (Module.rank F E) 1)
  -/
  rw [← rank_top, Subalgebra.rank_eq_one_iff, eq_comm]
  /-
    🎉 no goals
  -/


theorem bot_eq_top_iff_finrank_eq_one [Nontrivial E] [Module.Free F E] :
    (⊥ : Subalgebra F E) = ⊤ ↔ finrank F E = 1 := by
  /-
    F : Type u_1
    E : Type u_2
    inst✝⁵ : CommRing F
    inst✝⁴ : StrongRankCondition F
    inst✝³ : Ring E
    inst✝² : Algebra F E
    inst✝¹ : Nontrivial E
    inst✝ : Module.Free F E
    ⊢ Iff (Eq Bot.bot Top.top) (Eq (Module.finrank F E) 1)
  -/
  haveI := Module.Free.of_equiv (Subalgebra.topEquiv (R := F) (A := E)).toLinearEquiv.symm
  rw [← finrank_top, ← subalgebra_top_finrank_eq_submodule_top_finrank,
    Subalgebra.finrank_eq_one_iff, eq_comm]


alias ⟨_, bot_eq_top_of_rank_eq_one⟩ := bot_eq_top_iff_rank_eq_one


alias ⟨_, bot_eq_top_of_finrank_eq_one⟩ := bot_eq_top_iff_finrank_eq_one


