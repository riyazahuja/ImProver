/-- Meromorphy of `f` at `x` (more precisely, on a punctured neighbourhood of `x`; the value at
`x` itself is irrelevant). -/
def MeromorphicAt (f : 𝕜 → E) (x : 𝕜) :=
  ∃ (n : ℕ), AnalyticAt 𝕜 (fun z ↦ (z - x) ^ n • f z) x


lemma AnalyticAt.meromorphicAt {f : 𝕜 → E} {x : 𝕜} (hf : AnalyticAt 𝕜 f x) :
    MeromorphicAt f x :=
         /-
           𝕜 : Type u_1
           inst✝² : NontriviallyNormedField 𝕜
           E : Type u_2
           inst✝¹ : NormedAddCommGroup E
           inst✝ : NormedSpace 𝕜 E
           f : 𝕜 → E
           x : 𝕜
           hf : AnalyticAt 𝕜 f x
           ⊢ AnalyticAt 𝕜 (fun z => HSMul.hSMul (HPow.hPow (HSub.hSub z x) 0) (f z)) x
         -/
  ⟨0, by simpa only [pow_zero, one_smul]⟩
         /-
           🎉 no goals
         -/


lemma id (x : 𝕜) : MeromorphicAt id x := analyticAt_id.meromorphicAt


lemma const (e : E) (x : 𝕜) : MeromorphicAt (fun _ ↦ e) x :=
  analyticAt_const.meromorphicAt


lemma add {f g : 𝕜 → E} {x : 𝕜} (hf : MeromorphicAt f x) (hg : MeromorphicAt g x) :
    MeromorphicAt (f + g) x := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f g : 𝕜 → E
    x : 𝕜
    hf : MeromorphicAt f x
    hg : MeromorphicAt g x
    ⊢ MeromorphicAt (HAdd.hAdd f g) x
  -/
  rcases hf with ⟨m, hf⟩
  /-
    case intro
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f g : 𝕜 → E
    x : 𝕜
    hg : MeromorphicAt g x
    m : Nat
    hf : AnalyticAt 𝕜 (fun z => HSMul.hSMul (HPow.hPow (HSub.hSub z x) m) (f z)) x
    ⊢ MeromorphicAt (HAdd.hAdd f g) x
  -/
  rcases hg with ⟨n, hg⟩
  /-
    case intro.intro
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f g : 𝕜 → E
    x : 𝕜
    m : Nat
    hf : AnalyticAt 𝕜 (fun z => HSMul.hSMul (HPow.hPow (HSub.hSub z x) m) (f z)) x
    n : Nat
    hg : AnalyticAt 𝕜 (fun z => HSMul.hSMul (HPow.hPow (HSub.hSub z x) n) (g z)) x
    ⊢ MeromorphicAt (HAdd.hAdd f g) x
  -/
  refine ⟨max m n, ?_⟩
  have : (fun z ↦ (z - x) ^ max m n • (f + g) z) = fun z ↦ (z - x) ^ (max m n - m) •
      ((z - x) ^ m • f z) + (z - x) ^ (max m n - n) • ((z - x) ^ n • g z) := by
    simp_rw [← mul_smul, ← pow_add, Nat.sub_add_cancel (Nat.le_max_left _ _),
      Nat.sub_add_cancel (Nat.le_max_right _ _), Pi.add_apply, smul_add]
  /-
    case intro.intro
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f g : 𝕜 → E
    x : 𝕜
    m : Nat
    hf : AnalyticAt 𝕜 (fun z => HSMul.hSMul (HPow.hPow (HSub.hSub z x) m) (f z)) x
    n : Nat
    hg : AnalyticAt 𝕜 (fun z => HSMul.hSMul (HPow.hPow (HSub.hSub z x) n) (g z)) x
    this : Eq (fun z => HSMul.hSMul (HPow.hPow (HSub.hSub z x) (Max.max m n)) (HAd …
    ⊢ AnalyticAt 𝕜 (fun z => HSMul.hSMul (HPow.hPow (HSub.hSub z x) (Max.max m n)) …
  -/
  rw [this]
  exact (((analyticAt_id.sub analyticAt_const).pow _).smul hf).add
   (((analyticAt_id.sub analyticAt_const).pow _).smul hg)


lemma smul {f : 𝕜 → 𝕜} {g : 𝕜 → E} {x : 𝕜} (hf : MeromorphicAt f x) (hg : MeromorphicAt g x) :
    MeromorphicAt (f • g) x := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : 𝕜 → 𝕜
    g : 𝕜 → E
    x : 𝕜
    hf : MeromorphicAt f x
    hg : MeromorphicAt g x
    ⊢ MeromorphicAt (HSMul.hSMul f g) x
  -/
  rcases hf with ⟨m, hf⟩
  /-
    case intro
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : 𝕜 → 𝕜
    g : 𝕜 → E
    x : 𝕜
    hg : MeromorphicAt g x
    m : Nat
    hf : AnalyticAt 𝕜 (fun z => HSMul.hSMul (HPow.hPow (HSub.hSub z x) m) (f z)) x
    ⊢ MeromorphicAt (HSMul.hSMul f g) x
  -/
  rcases hg with ⟨n, hg⟩
  /-
    case intro.intro
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : 𝕜 → 𝕜
    g : 𝕜 → E
    x : 𝕜
    m : Nat
    hf : AnalyticAt 𝕜 (fun z => HSMul.hSMul (HPow.hPow (HSub.hSub z x) m) (f z)) x
    n : Nat
    hg : AnalyticAt 𝕜 (fun z => HSMul.hSMul (HPow.hPow (HSub.hSub z x) n) (g z)) x
    ⊢ MeromorphicAt (HSMul.hSMul f g) x
  -/
  refine ⟨m + n, ?_⟩
  /-
    case intro.intro
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : 𝕜 → 𝕜
    g : 𝕜 → E
    x : 𝕜
    m : Nat
    hf : AnalyticAt 𝕜 (fun z => HSMul.hSMul (HPow.hPow (HSub.hSub z x) m) (f z)) x
    n : Nat
    hg : AnalyticAt 𝕜 (fun z => HSMul.hSMul (HPow.hPow (HSub.hSub z x) n) (g z)) x
    ⊢ AnalyticAt 𝕜 (fun z => HSMul.hSMul (HPow.hPow (HSub.hSub z x) (HAdd.hAdd m n …
  -/
  convert hf.smul hg using 2 with z
  /-
    case h.e'_9.h
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : 𝕜 → 𝕜
    g : 𝕜 → E
    x : 𝕜
    m : Nat
    hf : AnalyticAt 𝕜 (fun z => HSMul.hSMul (HPow.hPow (HSub.hSub z x) m) (f z)) x
    n : Nat
    hg : AnalyticAt 𝕜 (fun z => HSMul.hSMul (HPow.hPow (HSub.hSub z x) n) (g z)) x
    z : 𝕜
    ⊢ Eq (HSMul.hSMul (HPow.hPow (HSub.hSub z x) (HAdd.hAdd m n)) (HSMul.hSMul f g …
  -/
  rw [Pi.smul_apply', smul_eq_mul]
  /-
    case h.e'_9.h
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : 𝕜 → 𝕜
    g : 𝕜 → E
    x : 𝕜
    m : Nat
    hf : AnalyticAt 𝕜 (fun z => HSMul.hSMul (HPow.hPow (HSub.hSub z x) m) (f z)) x
    n : Nat
    hg : AnalyticAt 𝕜 (fun z => HSMul.hSMul (HPow.hPow (HSub.hSub z x) n) (g z)) x
    z : 𝕜
    ⊢ Eq (HSMul.hSMul (HPow.hPow (HSub.hSub z x) (HAdd.hAdd m n)) (HSMul.hSMul (f  …
  -/
  module
  /-
    🎉 no goals
  -/


lemma mul {f g : 𝕜 → 𝕜} {x : 𝕜} (hf : MeromorphicAt f x) (hg : MeromorphicAt g x) :
    MeromorphicAt (f * g) x :=
  hf.smul hg


lemma neg {f : 𝕜 → E} {x : 𝕜} (hf : MeromorphicAt f x) : MeromorphicAt (-f) x := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : 𝕜 → E
    x : 𝕜
    hf : MeromorphicAt f x
    ⊢ MeromorphicAt (Neg.neg f) x
  -/
  convert (MeromorphicAt.const (-1 : 𝕜) x).smul hf using 1
  /-
    case h.e'_6
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : 𝕜 → E
    x : 𝕜
    hf : MeromorphicAt f x
    ⊢ Eq (Neg.neg f) (HSMul.hSMul (fun x => -1) f)
  -/
  ext1 z
  /-
    case h.e'_6.h
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : 𝕜 → E
    x : 𝕜
    hf : MeromorphicAt f x
    z : 𝕜
    ⊢ Eq (Neg.neg f z) (HSMul.hSMul (fun x => -1) f z)
  -/
  simp only [Pi.neg_apply, Pi.smul_apply', neg_smul, one_smul]
  /-
    🎉 no goals
  -/


@[simp]
lemma neg_iff {f : 𝕜 → E} {x : 𝕜} :
    MeromorphicAt (-f) x ↔ MeromorphicAt f x :=
              /-
                𝕜 : Type u_1
                inst✝² : NontriviallyNormedField 𝕜
                E : Type u_2
                inst✝¹ : NormedAddCommGroup E
                inst✝ : NormedSpace 𝕜 E
                f : 𝕜 → E
                x : 𝕜
                h : MeromorphicAt (Neg.neg f) x
                ⊢ MeromorphicAt f x
              -/
  ⟨fun h ↦ by simpa only [neg_neg] using h.neg, MeromorphicAt.neg⟩
              /-
                🎉 no goals
              -/


lemma sub {f g : 𝕜 → E} {x : 𝕜} (hf : MeromorphicAt f x) (hg : MeromorphicAt g x) :
    MeromorphicAt (f - g) x := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f g : 𝕜 → E
    x : 𝕜
    hf : MeromorphicAt f x
    hg : MeromorphicAt g x
    ⊢ MeromorphicAt (HSub.hSub f g) x
  -/
  convert hf.add hg.neg using 1
  /-
    case h.e'_6
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f g : 𝕜 → E
    x : 𝕜
    hf : MeromorphicAt f x
    hg : MeromorphicAt g x
    ⊢ Eq (HSub.hSub f g) (HAdd.hAdd f (Neg.neg g))
  -/
  ext1 z
  /-
    case h.e'_6.h
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f g : 𝕜 → E
    x : 𝕜
    hf : MeromorphicAt f x
    hg : MeromorphicAt g x
    z : 𝕜
    ⊢ Eq (HSub.hSub f g z) (HAdd.hAdd f (Neg.neg g) z)
  -/
  simp_rw [Pi.sub_apply, Pi.add_apply, Pi.neg_apply, sub_eq_add_neg]
  /-
    🎉 no goals
  -/


/-- With our definitions, `MeromorphicAt f x` depends only on the values of `f` on a punctured
neighbourhood of `x` (not on `f x`) -/
lemma congr {f g : 𝕜 → E} {x : 𝕜} (hf : MeromorphicAt f x) (hfg : f =ᶠ[𝓝[≠] x] g) :
    MeromorphicAt g x := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f g : 𝕜 → E
    x : 𝕜
    hf : MeromorphicAt f x
    hfg : (nhdsWithin x (HasCompl.compl (Singleton.singleton x))).EventuallyEq f g
    ⊢ MeromorphicAt g x
  -/
  rcases hf with ⟨m, hf⟩
  /-
    case intro
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f g : 𝕜 → E
    x : 𝕜
    hfg : (nhdsWithin x (HasCompl.compl (Singleton.singleton x))).EventuallyEq f g
    m : Nat
    hf : AnalyticAt 𝕜 (fun z => HSMul.hSMul (HPow.hPow (HSub.hSub z x) m) (f z)) x
    ⊢ MeromorphicAt g x
  -/
  refine ⟨m + 1, ?_⟩
  /-
    case intro
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f g : 𝕜 → E
    x : 𝕜
    hfg : (nhdsWithin x (HasCompl.compl (Singleton.singleton x))).EventuallyEq f g
    m : Nat
    hf : AnalyticAt 𝕜 (fun z => HSMul.hSMul (HPow.hPow (HSub.hSub z x) m) (f z)) x
    ⊢ AnalyticAt 𝕜 (fun z => HSMul.hSMul (HPow.hPow (HSub.hSub z x) (HAdd.hAdd m 1 …
  -/
  have : AnalyticAt 𝕜 (fun z ↦ z - x) x := analyticAt_id.sub analyticAt_const
  /-
    case intro
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f g : 𝕜 → E
    x : 𝕜
    hfg : (nhdsWithin x (HasCompl.compl (Singleton.singleton x))).EventuallyEq f g
    m : Nat
    hf : AnalyticAt 𝕜 (fun z => HSMul.hSMul (HPow.hPow (HSub.hSub z x) m) (f z)) x
    this : AnalyticAt 𝕜 (fun z => HSub.hSub z x) x
    ⊢ AnalyticAt 𝕜 (fun z => HSMul.hSMul (HPow.hPow (HSub.hSub z x) (HAdd.hAdd m 1 …
  -/
  refine (this.smul hf).congr ?_
  /-
    case intro
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f g : 𝕜 → E
    x : 𝕜
    hfg : (nhdsWithin x (HasCompl.compl (Singleton.singleton x))).EventuallyEq f g
    m : Nat
    hf : AnalyticAt 𝕜 (fun z => HSMul.hSMul (HPow.hPow (HSub.hSub z x) m) (f z)) x
    this : AnalyticAt 𝕜 (fun z => HSub.hSub z x) x
    ⊢ (nhds x).EventuallyEq (fun x_1 => HSMul.hSMul (HSub.hSub x_1 x) (HSMul.hSMul …
  -/
  rw [eventuallyEq_nhdsWithin_iff] at hfg
  /-
    case intro
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f g : 𝕜 → E
    x : 𝕜
    hfg : Filter.Eventually (fun x_1 => Membership.mem (HasCompl.compl (Singleton. …
    m : Nat
    hf : AnalyticAt 𝕜 (fun z => HSMul.hSMul (HPow.hPow (HSub.hSub z x) m) (f z)) x
    this : AnalyticAt 𝕜 (fun z => HSub.hSub z x) x
    ⊢ (nhds x).EventuallyEq (fun x_1 => HSMul.hSMul (HSub.hSub x_1 x) (HSMul.hSMul …
  -/
  filter_upwards [hfg] with z hz
  /-
    case h
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f g : 𝕜 → E
    x : 𝕜
    hfg : Filter.Eventually (fun x_1 => Membership.mem (HasCompl.compl (Singleton. …
    m : Nat
    hf : AnalyticAt 𝕜 (fun z => HSMul.hSMul (HPow.hPow (HSub.hSub z x) m) (f z)) x
    this : AnalyticAt 𝕜 (fun z => HSub.hSub z x) x
    z : 𝕜
    hz : Membership.mem (HasCompl.compl (Singleton.singleton x)) z → Eq (f z) (g z)
    ⊢ Eq (HSMul.hSMul (HSub.hSub z x) (HSMul.hSMul (HPow.hPow (HSub.hSub z x) m) ( …
  -/
  rcases eq_or_ne z x with rfl | hn
    /-
      case h.inl
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      f g : 𝕜 → E
      m : Nat
      z : 𝕜
      hfg : Filter.Eventually (fun x => Membership.mem (HasCompl.compl (Singleton.si …
      hf : AnalyticAt 𝕜 (fun z_1 => HSMul.hSMul (HPow.hPow (HSub.hSub z_1 z) m) (f z …
      this : AnalyticAt 𝕜 (fun z_1 => HSub.hSub z_1 z) z
      hz : Membership.mem (HasCompl.compl (Singleton.singleton z)) z → Eq (f z) (g z)
      ⊢ Eq (HSMul.hSMul (HSub.hSub z z) (HSMul.hSMul (HPow.hPow (HSub.hSub z z) m) ( …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case h.inr
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      f g : 𝕜 → E
      x : 𝕜
      hfg : Filter.Eventually (fun x_1 => Membership.mem (HasCompl.compl (Singleton. …
      m : Nat
      hf : AnalyticAt 𝕜 (fun z => HSMul.hSMul (HPow.hPow (HSub.hSub z x) m) (f z)) x
      this : AnalyticAt 𝕜 (fun z => HSub.hSub z x) x
      z : 𝕜
      hz : Membership.mem (HasCompl.compl (Singleton.singleton x)) z → Eq (f z) (g z)
      hn : Ne z x
      ⊢ Eq (HSMul.hSMul (HSub.hSub z x) (HSMul.hSMul (HPow.hPow (HSub.hSub z x) m) ( …
    -/
  · rw [hz (Set.mem_compl_singleton_iff.mp hn), pow_succ', mul_smul]
    /-
      🎉 no goals
    -/


lemma inv {f : 𝕜 → 𝕜} {x : 𝕜} (hf : MeromorphicAt f x) : MeromorphicAt f⁻¹ x := by
  /-
    𝕜 : Type u_1
    inst✝ : NontriviallyNormedField 𝕜
    f : 𝕜 → 𝕜
    x : 𝕜
    hf : MeromorphicAt f x
    ⊢ MeromorphicAt (Inv.inv f) x
  -/
  rcases hf with ⟨m, hf⟩
  /-
    case intro
    𝕜 : Type u_1
    inst✝ : NontriviallyNormedField 𝕜
    f : 𝕜 → 𝕜
    x : 𝕜
    m : Nat
    hf : AnalyticAt 𝕜 (fun z => HSMul.hSMul (HPow.hPow (HSub.hSub z x) m) (f z)) x
    ⊢ MeromorphicAt (Inv.inv f) x
  -/
  by_cases h_eq : (fun z ↦ (z - x) ^ m • f z) =ᶠ[𝓝 x] 0
  · -- silly case: f locally 0 near x
    /-
      case pos
      𝕜 : Type u_1
      inst✝ : NontriviallyNormedField 𝕜
      f : 𝕜 → 𝕜
      x : 𝕜
      m : Nat
      hf : AnalyticAt 𝕜 (fun z => HSMul.hSMul (HPow.hPow (HSub.hSub z x) m) (f z)) x
      h_eq : (nhds x).EventuallyEq (fun z => HSMul.hSMul (HPow.hPow (HSub.hSub z x)  …
      ⊢ MeromorphicAt (Inv.inv f) x
    -/
    refine (MeromorphicAt.const 0 x).congr ?_
    /-
      case pos
      𝕜 : Type u_1
      inst✝ : NontriviallyNormedField 𝕜
      f : 𝕜 → 𝕜
      x : 𝕜
      m : Nat
      hf : AnalyticAt 𝕜 (fun z => HSMul.hSMul (HPow.hPow (HSub.hSub z x) m) (f z)) x
      h_eq : (nhds x).EventuallyEq (fun z => HSMul.hSMul (HPow.hPow (HSub.hSub z x)  …
      ⊢ (nhdsWithin x (HasCompl.compl (Singleton.singleton x))).EventuallyEq (fun x  …
    -/
    rw [eventuallyEq_nhdsWithin_iff]
    /-
      case pos
      𝕜 : Type u_1
      inst✝ : NontriviallyNormedField 𝕜
      f : 𝕜 → 𝕜
      x : 𝕜
      m : Nat
      hf : AnalyticAt 𝕜 (fun z => HSMul.hSMul (HPow.hPow (HSub.hSub z x) m) (f z)) x
      h_eq : (nhds x).EventuallyEq (fun z => HSMul.hSMul (HPow.hPow (HSub.hSub z x)  …
      ⊢ Filter.Eventually (fun x_1 => Membership.mem (HasCompl.compl (Singleton.sing …
    -/
    filter_upwards [h_eq] with z hfz hz
    rw [Pi.inv_apply, (smul_eq_zero_iff_right <| pow_ne_zero _ (sub_ne_zero.mpr hz)).mp hfz,
      inv_zero]
  · -- interesting case: use local formula for `f`
    /-
      case neg
      𝕜 : Type u_1
      inst✝ : NontriviallyNormedField 𝕜
      f : 𝕜 → 𝕜
      x : 𝕜
      m : Nat
      hf : AnalyticAt 𝕜 (fun z => HSMul.hSMul (HPow.hPow (HSub.hSub z x) m) (f z)) x
      h_eq : Not ((nhds x).EventuallyEq (fun z => HSMul.hSMul (HPow.hPow (HSub.hSub  …
      ⊢ MeromorphicAt (Inv.inv f) x
    -/
    obtain ⟨n, g, hg_an, hg_ne, hg_eq⟩ := hf.exists_eventuallyEq_pow_smul_nonzero_iff.mpr h_eq
    have : AnalyticAt 𝕜 (fun z ↦ (z - x) ^ (m + 1)) x :=
      (analyticAt_id.sub analyticAt_const).pow _
    -- use `m + 1` rather than `m` to damp out any silly issues with the value at `z = x`
    /-
      case neg.intro.intro.intro.intro
      𝕜 : Type u_1
      inst✝ : NontriviallyNormedField 𝕜
      f : 𝕜 → 𝕜
      x : 𝕜
      m : Nat
      hf : AnalyticAt 𝕜 (fun z => HSMul.hSMul (HPow.hPow (HSub.hSub z x) m) (f z)) x
      h_eq : Not ((nhds x).EventuallyEq (fun z => HSMul.hSMul (HPow.hPow (HSub.hSub  …
      n : Nat
      g : 𝕜 → 𝕜
      hg_an : AnalyticAt 𝕜 g x
      hg_ne : Ne (g x) 0
      hg_eq : Filter.Eventually (fun z => Eq (HSMul.hSMul (HPow.hPow (HSub.hSub z x) …
      this : AnalyticAt 𝕜 (fun z => HPow.hPow (HSub.hSub z x) (HAdd.hAdd m 1)) x
      ⊢ MeromorphicAt (Inv.inv f) x
    -/
    refine ⟨n + 1, (this.smul <| hg_an.inv hg_ne).congr ?_⟩
    /-
      case neg.intro.intro.intro.intro
      𝕜 : Type u_1
      inst✝ : NontriviallyNormedField 𝕜
      f : 𝕜 → 𝕜
      x : 𝕜
      m : Nat
      hf : AnalyticAt 𝕜 (fun z => HSMul.hSMul (HPow.hPow (HSub.hSub z x) m) (f z)) x
      h_eq : Not ((nhds x).EventuallyEq (fun z => HSMul.hSMul (HPow.hPow (HSub.hSub  …
      n : Nat
      g : 𝕜 → 𝕜
      hg_an : AnalyticAt 𝕜 g x
      hg_ne : Ne (g x) 0
      hg_eq : Filter.Eventually (fun z => Eq (HSMul.hSMul (HPow.hPow (HSub.hSub z x) …
      this : AnalyticAt 𝕜 (fun z => HPow.hPow (HSub.hSub z x) (HAdd.hAdd m 1)) x
      ⊢ (nhds x).EventuallyEq (fun x_1 => HSMul.hSMul (HPow.hPow (HSub.hSub x_1 x) ( …
    -/
    filter_upwards [hg_eq, hg_an.continuousAt.eventually_ne hg_ne] with z hfg hg_ne'
    /-
      case h
      𝕜 : Type u_1
      inst✝ : NontriviallyNormedField 𝕜
      f : 𝕜 → 𝕜
      x : 𝕜
      m : Nat
      hf : AnalyticAt 𝕜 (fun z => HSMul.hSMul (HPow.hPow (HSub.hSub z x) m) (f z)) x
      h_eq : Not ((nhds x).EventuallyEq (fun z => HSMul.hSMul (HPow.hPow (HSub.hSub  …
      n : Nat
      g : 𝕜 → 𝕜
      hg_an : AnalyticAt 𝕜 g x
      hg_ne : Ne (g x) 0
      hg_eq : Filter.Eventually (fun z => Eq (HSMul.hSMul (HPow.hPow (HSub.hSub z x) …
      this : AnalyticAt 𝕜 (fun z => HPow.hPow (HSub.hSub z x) (HAdd.hAdd m 1)) x
      z : 𝕜
      hfg : Eq (HSMul.hSMul (HPow.hPow (HSub.hSub z x) m) (f z)) (HSMul.hSMul (HPow. …
      hg_ne' : Ne (g z) 0
      ⊢ Eq (HSMul.hSMul (HPow.hPow (HSub.hSub z x) (HAdd.hAdd m 1)) (Inv.inv (g z))) …
    -/
    rcases eq_or_ne z x with rfl | hz_ne
      /-
        case h.inl
        𝕜 : Type u_1
        inst✝ : NontriviallyNormedField 𝕜
        f : 𝕜 → 𝕜
        m n : Nat
        g : 𝕜 → 𝕜
        z : 𝕜
        hg_ne' : Ne (g z) 0
        hf : AnalyticAt 𝕜 (fun z_1 => HSMul.hSMul (HPow.hPow (HSub.hSub z_1 z) m) (f z …
        h_eq : Not ((nhds z).EventuallyEq (fun z_1 => HSMul.hSMul (HPow.hPow (HSub.hSu …
        hg_an : AnalyticAt 𝕜 g z
        hg_ne : Ne (g z) 0
        hg_eq : Filter.Eventually (fun z_1 => Eq (HSMul.hSMul (HPow.hPow (HSub.hSub z_ …
        this : AnalyticAt 𝕜 (fun z_1 => HPow.hPow (HSub.hSub z_1 z) (HAdd.hAdd m 1)) z
        hfg : Eq (HSMul.hSMul (HPow.hPow (HSub.hSub z z) m) (f z)) (HSMul.hSMul (HPow. …
        ⊢ Eq (HSMul.hSMul (HPow.hPow (HSub.hSub z z) (HAdd.hAdd m 1)) (Inv.inv (g z))) …
      -/
    · simp only [sub_self, pow_succ, mul_zero, zero_smul]
      /-
        🎉 no goals
      -/
      /-
        case h.inr
        𝕜 : Type u_1
        inst✝ : NontriviallyNormedField 𝕜
        f : 𝕜 → 𝕜
        x : 𝕜
        m : Nat
        hf : AnalyticAt 𝕜 (fun z => HSMul.hSMul (HPow.hPow (HSub.hSub z x) m) (f z)) x
        h_eq : Not ((nhds x).EventuallyEq (fun z => HSMul.hSMul (HPow.hPow (HSub.hSub  …
        n : Nat
        g : 𝕜 → 𝕜
        hg_an : AnalyticAt 𝕜 g x
        hg_ne : Ne (g x) 0
        hg_eq : Filter.Eventually (fun z => Eq (HSMul.hSMul (HPow.hPow (HSub.hSub z x) …
        this : AnalyticAt 𝕜 (fun z => HPow.hPow (HSub.hSub z x) (HAdd.hAdd m 1)) x
        z : 𝕜
        hfg : Eq (HSMul.hSMul (HPow.hPow (HSub.hSub z x) m) (f z)) (HSMul.hSMul (HPow. …
        hg_ne' : Ne (g z) 0
        hz_ne : Ne z x
        ⊢ Eq (HSMul.hSMul (HPow.hPow (HSub.hSub z x) (HAdd.hAdd m 1)) (Inv.inv (g z))) …
      -/
    · simp_rw [smul_eq_mul] at hfg ⊢
      have aux1 : f z ≠ 0 := by
        have : (z - x) ^ n * g z ≠ 0 := mul_ne_zero (pow_ne_zero _ (sub_ne_zero.mpr hz_ne)) hg_ne'
        rw [← hfg, mul_ne_zero_iff] at this
        exact this.2
      /-
        case h.inr
        𝕜 : Type u_1
        inst✝ : NontriviallyNormedField 𝕜
        f : 𝕜 → 𝕜
        x : 𝕜
        m : Nat
        hf : AnalyticAt 𝕜 (fun z => HSMul.hSMul (HPow.hPow (HSub.hSub z x) m) (f z)) x
        h_eq : Not ((nhds x).EventuallyEq (fun z => HSMul.hSMul (HPow.hPow (HSub.hSub  …
        n : Nat
        g : 𝕜 → 𝕜
        hg_an : AnalyticAt 𝕜 g x
        hg_ne : Ne (g x) 0
        hg_eq : Filter.Eventually (fun z => Eq (HSMul.hSMul (HPow.hPow (HSub.hSub z x) …
        this : AnalyticAt 𝕜 (fun z => HPow.hPow (HSub.hSub z x) (HAdd.hAdd m 1)) x
        z : 𝕜
        hfg : Eq (HMul.hMul (HPow.hPow (HSub.hSub z x) m) (f z)) (HMul.hMul (HPow.hPow …
        hg_ne' : Ne (g z) 0
        hz_ne : Ne z x
        aux1 : Ne (f z) 0
        ⊢ Eq (HMul.hMul (HPow.hPow (HSub.hSub z x) (HAdd.hAdd m 1)) (Inv.inv (g z))) ( …
      -/
      field_simp [sub_ne_zero.mpr hz_ne]
      /-
        case h.inr
        𝕜 : Type u_1
        inst✝ : NontriviallyNormedField 𝕜
        f : 𝕜 → 𝕜
        x : 𝕜
        m : Nat
        hf : AnalyticAt 𝕜 (fun z => HSMul.hSMul (HPow.hPow (HSub.hSub z x) m) (f z)) x
        h_eq : Not ((nhds x).EventuallyEq (fun z => HSMul.hSMul (HPow.hPow (HSub.hSub  …
        n : Nat
        g : 𝕜 → 𝕜
        hg_an : AnalyticAt 𝕜 g x
        hg_ne : Ne (g x) 0
        hg_eq : Filter.Eventually (fun z => Eq (HSMul.hSMul (HPow.hPow (HSub.hSub z x) …
        this : AnalyticAt 𝕜 (fun z => HPow.hPow (HSub.hSub z x) (HAdd.hAdd m 1)) x
        z : 𝕜
        hfg : Eq (HMul.hMul (HPow.hPow (HSub.hSub z x) m) (f z)) (HMul.hMul (HPow.hPow …
        hg_ne' : Ne (g z) 0
        hz_ne : Ne z x
        aux1 : Ne (f z) 0
        ⊢ Eq (HMul.hMul (HPow.hPow (HSub.hSub z x) (HAdd.hAdd m 1)) (f z)) (HMul.hMul  …
      -/
      rw [pow_succ', mul_assoc, hfg]
      /-
        case h.inr
        𝕜 : Type u_1
        inst✝ : NontriviallyNormedField 𝕜
        f : 𝕜 → 𝕜
        x : 𝕜
        m : Nat
        hf : AnalyticAt 𝕜 (fun z => HSMul.hSMul (HPow.hPow (HSub.hSub z x) m) (f z)) x
        h_eq : Not ((nhds x).EventuallyEq (fun z => HSMul.hSMul (HPow.hPow (HSub.hSub  …
        n : Nat
        g : 𝕜 → 𝕜
        hg_an : AnalyticAt 𝕜 g x
        hg_ne : Ne (g x) 0
        hg_eq : Filter.Eventually (fun z => Eq (HSMul.hSMul (HPow.hPow (HSub.hSub z x) …
        this : AnalyticAt 𝕜 (fun z => HPow.hPow (HSub.hSub z x) (HAdd.hAdd m 1)) x
        z : 𝕜
        hfg : Eq (HMul.hMul (HPow.hPow (HSub.hSub z x) m) (f z)) (HMul.hMul (HPow.hPow …
        hg_ne' : Ne (g z) 0
        hz_ne : Ne z x
        aux1 : Ne (f z) 0
        ⊢ Eq (HMul.hMul (HSub.hSub z x) (HMul.hMul (HPow.hPow (HSub.hSub z x) n) (g z) …
      -/
      ring
      /-
        🎉 no goals
      -/


@[simp]
lemma inv_iff {f : 𝕜 → 𝕜} {x : 𝕜} :
    MeromorphicAt f⁻¹ x ↔ MeromorphicAt f x :=
              /-
                𝕜 : Type u_1
                inst✝ : NontriviallyNormedField 𝕜
                f : 𝕜 → 𝕜
                x : 𝕜
                h : MeromorphicAt (Inv.inv f) x
                ⊢ MeromorphicAt f x
              -/
  ⟨fun h ↦ by simpa only [inv_inv] using h.inv, MeromorphicAt.inv⟩
              /-
                🎉 no goals
              -/


lemma div {f g : 𝕜 → 𝕜} {x : 𝕜} (hf : MeromorphicAt f x) (hg : MeromorphicAt g x) :
    MeromorphicAt (f / g) x :=
  (div_eq_mul_inv f g).symm ▸ (hf.mul hg.inv)


lemma pow {f : 𝕜 → 𝕜} {x : 𝕜} (hf : MeromorphicAt f x) (n : ℕ) : MeromorphicAt (f ^ n) x := by
  induction n with
  | zero => simpa only [pow_zero] using MeromorphicAt.const 1 x
  | succ m hm => simpa only [pow_succ] using hm.mul hf


lemma zpow {f : 𝕜 → 𝕜} {x : 𝕜} (hf : MeromorphicAt f x) (n : ℤ) : MeromorphicAt (f ^ n) x := by
  induction n with
  | ofNat m => simpa only [Int.ofNat_eq_coe, zpow_natCast] using hf.pow m
  | negSucc m => simpa only [zpow_negSucc, inv_iff] using hf.pow (m + 1)


theorem eventually_analyticAt [CompleteSpace E] {f : 𝕜 → E} {x : 𝕜}
    (h : MeromorphicAt f x) : ∀ᶠ y in 𝓝[≠] x, AnalyticAt 𝕜 f y := by
  /-
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : CompleteSpace E
    f : 𝕜 → E
    x : 𝕜
    h : MeromorphicAt f x
    ⊢ Filter.Eventually (fun y => AnalyticAt 𝕜 f y) (nhdsWithin x (HasCompl.compl  …
  -/
  rw [MeromorphicAt] at h
  /-
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : CompleteSpace E
    f : 𝕜 → E
    x : 𝕜
    h : Exists fun n => AnalyticAt 𝕜 (fun z => HSMul.hSMul (HPow.hPow (HSub.hSub z …
    ⊢ Filter.Eventually (fun y => AnalyticAt 𝕜 f y) (nhdsWithin x (HasCompl.compl  …
  -/
  obtain ⟨n, h⟩ := h
  /-
    case intro
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : CompleteSpace E
    f : 𝕜 → E
    x : 𝕜
    n : Nat
    h : AnalyticAt 𝕜 (fun z => HSMul.hSMul (HPow.hPow (HSub.hSub z x) n) (f z)) x
    ⊢ Filter.Eventually (fun y => AnalyticAt 𝕜 f y) (nhdsWithin x (HasCompl.compl  …
  -/
  apply AnalyticAt.eventually_analyticAt at h
  /-
    case intro
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : CompleteSpace E
    f : 𝕜 → E
    x : 𝕜
    n : Nat
    h : Filter.Eventually (fun y => AnalyticAt 𝕜 (fun z => HSMul.hSMul (HPow.hPow  …
    ⊢ Filter.Eventually (fun y => AnalyticAt 𝕜 f y) (nhdsWithin x (HasCompl.compl  …
  -/
  refine (h.filter_mono ?_).mp ?_
    /-
      case intro.refine_1
      𝕜 : Type u_1
      inst✝³ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace 𝕜 E
      inst✝ : CompleteSpace E
      f : 𝕜 → E
      x : 𝕜
      n : Nat
      h : Filter.Eventually (fun y => AnalyticAt 𝕜 (fun z => HSMul.hSMul (HPow.hPow  …
      ⊢ LE.le (nhdsWithin x (HasCompl.compl (Singleton.singleton x))) (nhds x)
    -/
  · simp [nhdsWithin]
    /-
      🎉 no goals
    -/
    /-
      case intro.refine_2
      𝕜 : Type u_1
      inst✝³ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace 𝕜 E
      inst✝ : CompleteSpace E
      f : 𝕜 → E
      x : 𝕜
      n : Nat
      h : Filter.Eventually (fun y => AnalyticAt 𝕜 (fun z => HSMul.hSMul (HPow.hPow  …
      ⊢ Filter.Eventually (fun x_1 => AnalyticAt 𝕜 (fun z => HSMul.hSMul (HPow.hPow  …
    -/
  · rw [eventually_nhdsWithin_iff]
    /-
      case intro.refine_2
      𝕜 : Type u_1
      inst✝³ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace 𝕜 E
      inst✝ : CompleteSpace E
      f : 𝕜 → E
      x : 𝕜
      n : Nat
      h : Filter.Eventually (fun y => AnalyticAt 𝕜 (fun z => HSMul.hSMul (HPow.hPow  …
      ⊢ Filter.Eventually (fun x_1 => Membership.mem (HasCompl.compl (Singleton.sing …
    -/
    apply Filter.Eventually.of_forall
    /-
      case intro.refine_2.hp
      𝕜 : Type u_1
      inst✝³ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace 𝕜 E
      inst✝ : CompleteSpace E
      f : 𝕜 → E
      x : 𝕜
      n : Nat
      h : Filter.Eventually (fun y => AnalyticAt 𝕜 (fun z => HSMul.hSMul (HPow.hPow  …
      ⊢ ∀ (x_1 : 𝕜), Membership.mem (HasCompl.compl (Singleton.singleton x)) x_1 → A …
    -/
    intro y hy hf
    /-
      case intro.refine_2.hp
      𝕜 : Type u_1
      inst✝³ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace 𝕜 E
      inst✝ : CompleteSpace E
      f : 𝕜 → E
      x : 𝕜
      n : Nat
      h : Filter.Eventually (fun y => AnalyticAt 𝕜 (fun z => HSMul.hSMul (HPow.hPow  …
      y : 𝕜
      hy : Membership.mem (HasCompl.compl (Singleton.singleton x)) y
      hf : AnalyticAt 𝕜 (fun z => HSMul.hSMul (HPow.hPow (HSub.hSub z x) n) (f z)) y
      ⊢ AnalyticAt 𝕜 f y
    -/
    rw [Set.mem_compl_iff, Set.mem_singleton_iff] at hy
    have := ((analyticAt_id (𝕜 := 𝕜).sub analyticAt_const).pow n).inv
      (pow_ne_zero _ (sub_ne_zero_of_ne hy))
    /-
      case intro.refine_2.hp
      𝕜 : Type u_1
      inst✝³ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace 𝕜 E
      inst✝ : CompleteSpace E
      f : 𝕜 → E
      x : 𝕜
      n : Nat
      h : Filter.Eventually (fun y => AnalyticAt 𝕜 (fun z => HSMul.hSMul (HPow.hPow  …
      y : 𝕜
      hy : Not (Eq y x)
      hf : AnalyticAt 𝕜 (fun z => HSMul.hSMul (HPow.hPow (HSub.hSub z x) n) (f z)) y
      this : AnalyticAt 𝕜 (fun x_1 => Inv.inv (HPow.hPow (HSub.hSub _root_.id (fun x …
      ⊢ AnalyticAt 𝕜 f y
    -/
    apply (this.smul hf).congr ∘ (eventually_ne_nhds hy).mono
    /-
      case intro.refine_2.hp
      𝕜 : Type u_1
      inst✝³ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace 𝕜 E
      inst✝ : CompleteSpace E
      f : 𝕜 → E
      x : 𝕜
      n : Nat
      h : Filter.Eventually (fun y => AnalyticAt 𝕜 (fun z => HSMul.hSMul (HPow.hPow  …
      y : 𝕜
      hy : Not (Eq y x)
      hf : AnalyticAt 𝕜 (fun z => HSMul.hSMul (HPow.hPow (HSub.hSub z x) n) (f z)) y
      this : AnalyticAt 𝕜 (fun x_1 => Inv.inv (HPow.hPow (HSub.hSub _root_.id (fun x …
      ⊢ ∀ (x_1 : 𝕜), Ne x_1 x → Eq ((fun x_2 => HSMul.hSMul (Inv.inv (HPow.hPow (HSu …
    -/
    intro z hz
    /-
      case intro.refine_2.hp
      𝕜 : Type u_1
      inst✝³ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace 𝕜 E
      inst✝ : CompleteSpace E
      f : 𝕜 → E
      x : 𝕜
      n : Nat
      h : Filter.Eventually (fun y => AnalyticAt 𝕜 (fun z => HSMul.hSMul (HPow.hPow  …
      y : 𝕜
      hy : Not (Eq y x)
      hf : AnalyticAt 𝕜 (fun z => HSMul.hSMul (HPow.hPow (HSub.hSub z x) n) (f z)) y
      this : AnalyticAt 𝕜 (fun x_1 => Inv.inv (HPow.hPow (HSub.hSub _root_.id (fun x …
      z : 𝕜
      hz : Ne z x
      ⊢ Eq ((fun x_1 => HSMul.hSMul (Inv.inv (HPow.hPow (HSub.hSub _root_.id (fun x_ …
    -/
    simp [smul_smul, hz, sub_eq_zero]
    /-
      🎉 no goals
    -/


/-- The order of vanishing of a meromorphic function, as an element of `ℤ ∪ ∞` (to include the
case of functions identically 0 near `x`). -/
noncomputable def order {f : 𝕜 → E} {x : 𝕜} (hf : MeromorphicAt f x) : WithTop ℤ :=
  (hf.choose_spec.order.map (↑· : ℕ → ℤ)) - hf.choose


lemma order_eq_top_iff {f : 𝕜 → E} {x : 𝕜} (hf : MeromorphicAt f x) :
    hf.order = ⊤ ↔ ∀ᶠ z in 𝓝[≠] x, f z = 0 := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : 𝕜 → E
    x : 𝕜
    hf : MeromorphicAt f x
    ⊢ Iff (Eq hf.order Top.top) (Filter.Eventually (fun z => Eq (f z) 0) (nhdsWith …
  -/
  unfold order
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : 𝕜 → E
    x : 𝕜
    hf : MeromorphicAt f x
    ⊢ Iff (Eq (HSub.hSub (ENat.map (fun x => ↑x) ⋯.order) ↑(Exists.choose hf)) Top …
  -/
  by_cases h : hf.choose_spec.order = ⊤
  · rw [h, ENat.map_top, ← WithTop.coe_natCast,
      top_sub, eq_self, true_iff, eventually_nhdsWithin_iff]
    /-
      case pos
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      f : 𝕜 → E
      x : 𝕜
      hf : MeromorphicAt f x
      h : Eq ⋯.order Top.top
      ⊢ Filter.Eventually (fun x_1 => Membership.mem (HasCompl.compl (Singleton.sing …
    -/
    rw [AnalyticAt.order_eq_top_iff] at h
    /-
      case pos
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      f : 𝕜 → E
      x : 𝕜
      hf : MeromorphicAt f x
      h : Filter.Eventually (fun z => Eq (HSMul.hSMul (HPow.hPow (HSub.hSub z x) (Ex …
      ⊢ Filter.Eventually (fun x_1 => Membership.mem (HasCompl.compl (Singleton.sing …
    -/
    filter_upwards [h] with z hf hz
    /-
      case h
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      f : 𝕜 → E
      x : 𝕜
      hf✝ : MeromorphicAt f x
      h : Filter.Eventually (fun z => Eq (HSMul.hSMul (HPow.hPow (HSub.hSub z x) (Ex …
      z : 𝕜
      hf : Eq (HSMul.hSMul (HPow.hPow (HSub.hSub z x) (Exists.choose hf✝)) (f z)) 0
      hz : Membership.mem (HasCompl.compl (Singleton.singleton x)) z
      ⊢ Eq (f z) 0
    -/
    rwa [smul_eq_zero_iff_right <| pow_ne_zero _ (sub_ne_zero.mpr hz)] at hf
    /-
      🎉 no goals
    -/
    /-
      case neg
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      f : 𝕜 → E
      x : 𝕜
      hf : MeromorphicAt f x
      h : Not (Eq ⋯.order Top.top)
      ⊢ Iff (Eq (HSub.hSub (ENat.map (fun x => ↑x) ⋯.order) ↑(Exists.choose hf)) Top …
    -/
  · obtain ⟨m, hm⟩ := ENat.ne_top_iff_exists.mp h
    simp only [← hm, ENat.map_coe, WithTop.coe_natCast, sub_eq_top_iff, WithTop.natCast_ne_top,
      or_self, false_iff]
    /-
      case neg.intro
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      f : 𝕜 → E
      x : 𝕜
      hf : MeromorphicAt f x
      h : Not (Eq ⋯.order Top.top)
      m : Nat
      hm : Eq ↑m ⋯.order
      ⊢ Not (Filter.Eventually (fun z => Eq (f z) 0) (nhdsWithin x (HasCompl.compl ( …
    -/
    contrapose! h
    /-
      case neg.intro
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      f : 𝕜 → E
      x : 𝕜
      hf : MeromorphicAt f x
      m : Nat
      hm : Eq ↑m ⋯.order
      h : Filter.Eventually (fun z => Eq (f z) 0) (nhdsWithin x (HasCompl.compl (Sin …
      ⊢ Eq ⋯.order Top.top
    -/
    rw [AnalyticAt.order_eq_top_iff]
    /-
      case neg.intro
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      f : 𝕜 → E
      x : 𝕜
      hf : MeromorphicAt f x
      m : Nat
      hm : Eq ↑m ⋯.order
      h : Filter.Eventually (fun z => Eq (f z) 0) (nhdsWithin x (HasCompl.compl (Sin …
      ⊢ Filter.Eventually (fun z => Eq (HSMul.hSMul (HPow.hPow (HSub.hSub z x) (Exis …
    -/
    rw [← hf.choose_spec.frequently_eq_iff_eventually_eq analyticAt_const]
    /-
      case neg.intro
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      f : 𝕜 → E
      x : 𝕜
      hf : MeromorphicAt f x
      m : Nat
      hm : Eq ↑m ⋯.order
      h : Filter.Eventually (fun z => Eq (f z) 0) (nhdsWithin x (HasCompl.compl (Sin …
      ⊢ Filter.Frequently (fun z => Eq (HSMul.hSMul (HPow.hPow (HSub.hSub z x) (Exis …
    -/
    apply Eventually.frequently
    /-
      case neg.intro.h
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      f : 𝕜 → E
      x : 𝕜
      hf : MeromorphicAt f x
      m : Nat
      hm : Eq ↑m ⋯.order
      h : Filter.Eventually (fun z => Eq (f z) 0) (nhdsWithin x (HasCompl.compl (Sin …
      ⊢ Filter.Eventually (fun x_1 => Eq (HSMul.hSMul (HPow.hPow (HSub.hSub x_1 x) ( …
    -/
    filter_upwards [h] with z hfz
    /-
      case h
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      f : 𝕜 → E
      x : 𝕜
      hf : MeromorphicAt f x
      m : Nat
      hm : Eq ↑m ⋯.order
      h : Filter.Eventually (fun z => Eq (f z) 0) (nhdsWithin x (HasCompl.compl (Sin …
      z : 𝕜
      hfz : Eq (f z) 0
      ⊢ Eq (HSMul.hSMul (HPow.hPow (HSub.hSub z x) (Exists.choose hf)) (f z)) 0
    -/
    rw [hfz, smul_zero]
    /-
      🎉 no goals
    -/


lemma order_eq_int_iff {f : 𝕜 → E} {x : 𝕜} (hf : MeromorphicAt f x) (n : ℤ) : hf.order = n ↔
    ∃ g : 𝕜 → E, AnalyticAt 𝕜 g x ∧ g x ≠ 0 ∧ ∀ᶠ z in 𝓝[≠] x, f z = (z - x) ^ n • g z := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : 𝕜 → E
    x : 𝕜
    hf : MeromorphicAt f x
    n : Int
    ⊢ Iff (Eq hf.order ↑n) (Exists fun g => And (AnalyticAt 𝕜 g x) (And (Ne (g x)  …
  -/
  unfold order
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : 𝕜 → E
    x : 𝕜
    hf : MeromorphicAt f x
    n : Int
    ⊢ Iff (Eq (HSub.hSub (ENat.map (fun x => ↑x) ⋯.order) ↑(Exists.choose hf)) ↑n) …
  -/
  by_cases h : hf.choose_spec.order = ⊤
  · rw [h, ENat.map_top, ← WithTop.coe_natCast, top_sub,
      eq_false_intro WithTop.top_ne_coe, false_iff]
    /-
      case pos
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      f : 𝕜 → E
      x : 𝕜
      hf : MeromorphicAt f x
      n : Int
      h : Eq ⋯.order Top.top
      ⊢ Not (Exists fun g => And (AnalyticAt 𝕜 g x) (And (Ne (g x) 0) (Filter.Eventu …
    -/
    rw [AnalyticAt.order_eq_top_iff] at h
    /-
      case pos
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      f : 𝕜 → E
      x : 𝕜
      hf : MeromorphicAt f x
      n : Int
      h : Filter.Eventually (fun z => Eq (HSMul.hSMul (HPow.hPow (HSub.hSub z x) (Ex …
      ⊢ Not (Exists fun g => And (AnalyticAt 𝕜 g x) (And (Ne (g x) 0) (Filter.Eventu …
    -/
    refine fun ⟨g, hg_an, hg_ne, hg_eq⟩ ↦ hg_ne ?_
    /-
      case pos
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      f : 𝕜 → E
      x : 𝕜
      hf : MeromorphicAt f x
      n : Int
      h : Filter.Eventually (fun z => Eq (HSMul.hSMul (HPow.hPow (HSub.hSub z x) (Ex …
      x✝ : Exists fun g => And (AnalyticAt 𝕜 g x) (And (Ne (g x) 0) (Filter.Eventual …
      g : 𝕜 → E
      hg_an : AnalyticAt 𝕜 g x
      hg_ne : Ne (g x) 0
      hg_eq : Filter.Eventually (fun z => Eq (f z) (HSMul.hSMul (HPow.hPow (HSub.hSu …
      ⊢ Eq (g x) 0
    -/
    apply EventuallyEq.eq_of_nhds
    /-
      case pos.h
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      f : 𝕜 → E
      x : 𝕜
      hf : MeromorphicAt f x
      n : Int
      h : Filter.Eventually (fun z => Eq (HSMul.hSMul (HPow.hPow (HSub.hSub z x) (Ex …
      x✝ : Exists fun g => And (AnalyticAt 𝕜 g x) (And (Ne (g x) 0) (Filter.Eventual …
      g : 𝕜 → E
      hg_an : AnalyticAt 𝕜 g x
      hg_ne : Ne (g x) 0
      hg_eq : Filter.Eventually (fun z => Eq (f z) (HSMul.hSMul (HPow.hPow (HSub.hSu …
      ⊢ (nhds x).EventuallyEq g fun {x} => 0
    -/
    rw [EventuallyEq, ← AnalyticAt.frequently_eq_iff_eventually_eq hg_an analyticAt_const]
    /-
      case pos.h
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      f : 𝕜 → E
      x : 𝕜
      hf : MeromorphicAt f x
      n : Int
      h : Filter.Eventually (fun z => Eq (HSMul.hSMul (HPow.hPow (HSub.hSub z x) (Ex …
      x✝ : Exists fun g => And (AnalyticAt 𝕜 g x) (And (Ne (g x) 0) (Filter.Eventual …
      g : 𝕜 → E
      hg_an : AnalyticAt 𝕜 g x
      hg_ne : Ne (g x) 0
      hg_eq : Filter.Eventually (fun z => Eq (f z) (HSMul.hSMul (HPow.hPow (HSub.hSu …
      ⊢ Filter.Frequently (fun z => Eq (g z) 0) (nhdsWithin x (HasCompl.compl (Singl …
    -/
    apply Eventually.frequently
    /-
      case pos.h.h
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      f : 𝕜 → E
      x : 𝕜
      hf : MeromorphicAt f x
      n : Int
      h : Filter.Eventually (fun z => Eq (HSMul.hSMul (HPow.hPow (HSub.hSub z x) (Ex …
      x✝ : Exists fun g => And (AnalyticAt 𝕜 g x) (And (Ne (g x) 0) (Filter.Eventual …
      g : 𝕜 → E
      hg_an : AnalyticAt 𝕜 g x
      hg_ne : Ne (g x) 0
      hg_eq : Filter.Eventually (fun z => Eq (f z) (HSMul.hSMul (HPow.hPow (HSub.hSu …
      ⊢ Filter.Eventually (fun x => Eq (g x) 0) (nhdsWithin x (HasCompl.compl (Singl …
    -/
    rw [eventually_nhdsWithin_iff] at hg_eq ⊢
    /-
      case pos.h.h
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      f : 𝕜 → E
      x : 𝕜
      hf : MeromorphicAt f x
      n : Int
      h : Filter.Eventually (fun z => Eq (HSMul.hSMul (HPow.hPow (HSub.hSub z x) (Ex …
      x✝ : Exists fun g => And (AnalyticAt 𝕜 g x) (And (Ne (g x) 0) (Filter.Eventual …
      g : 𝕜 → E
      hg_an : AnalyticAt 𝕜 g x
      hg_ne : Ne (g x) 0
      hg_eq : Filter.Eventually (fun x_1 => Membership.mem (HasCompl.compl (Singleto …
      ⊢ Filter.Eventually (fun x_1 => Membership.mem (HasCompl.compl (Singleton.sing …
    -/
    filter_upwards [h, hg_eq] with z hfz hfz_eq hz
    /-
      case h
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      f : 𝕜 → E
      x : 𝕜
      hf : MeromorphicAt f x
      n : Int
      h : Filter.Eventually (fun z => Eq (HSMul.hSMul (HPow.hPow (HSub.hSub z x) (Ex …
      x✝ : Exists fun g => And (AnalyticAt 𝕜 g x) (And (Ne (g x) 0) (Filter.Eventual …
      g : 𝕜 → E
      hg_an : AnalyticAt 𝕜 g x
      hg_ne : Ne (g x) 0
      hg_eq : Filter.Eventually (fun x_1 => Membership.mem (HasCompl.compl (Singleto …
      z : 𝕜
      hfz : Eq (HSMul.hSMul (HPow.hPow (HSub.hSub z x) (Exists.choose hf)) (f z)) 0
      hfz_eq : Membership.mem (HasCompl.compl (Singleton.singleton x)) z → Eq (f z)  …
      hz : Membership.mem (HasCompl.compl (Singleton.singleton x)) z
      ⊢ Eq (g z) 0
    -/
    rwa [hfz_eq hz, ← mul_smul, smul_eq_zero_iff_right] at hfz
    /-
      case h
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      f : 𝕜 → E
      x : 𝕜
      hf : MeromorphicAt f x
      n : Int
      h : Filter.Eventually (fun z => Eq (HSMul.hSMul (HPow.hPow (HSub.hSub z x) (Ex …
      x✝ : Exists fun g => And (AnalyticAt 𝕜 g x) (And (Ne (g x) 0) (Filter.Eventual …
      g : 𝕜 → E
      hg_an : AnalyticAt 𝕜 g x
      hg_ne : Ne (g x) 0
      hg_eq : Filter.Eventually (fun x_1 => Membership.mem (HasCompl.compl (Singleto …
      z : 𝕜
      hfz : Eq (HSMul.hSMul (HMul.hMul (HPow.hPow (HSub.hSub z x) (Exists.choose hf) …
      hfz_eq : Membership.mem (HasCompl.compl (Singleton.singleton x)) z → Eq (f z)  …
      hz : Membership.mem (HasCompl.compl (Singleton.singleton x)) z
      ⊢ Ne (HMul.hMul (HPow.hPow (HSub.hSub z x) (Exists.choose hf)) (HPow.hPow (HSu …
    -/
    exact mul_ne_zero (pow_ne_zero _ (sub_ne_zero.mpr hz)) (zpow_ne_zero _ (sub_ne_zero.mpr hz))
    /-
      🎉 no goals
    -/
    /-
      case neg
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      f : 𝕜 → E
      x : 𝕜
      hf : MeromorphicAt f x
      n : Int
      h : Not (Eq ⋯.order Top.top)
      ⊢ Iff (Eq (HSub.hSub (ENat.map (fun x => ↑x) ⋯.order) ↑(Exists.choose hf)) ↑n) …
    -/
  · obtain ⟨m, h⟩ := ENat.ne_top_iff_exists.mp h
    /-
      case neg.intro
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      f : 𝕜 → E
      x : 𝕜
      hf : MeromorphicAt f x
      n : Int
      h✝ : Not (Eq ⋯.order Top.top)
      m : Nat
      h : Eq ↑m ⋯.order
      ⊢ Iff (Eq (HSub.hSub (ENat.map (fun x => ↑x) ⋯.order) ↑(Exists.choose hf)) ↑n) …
    -/
    rw [← h, ENat.map_coe, ← WithTop.coe_natCast, ← coe_sub, WithTop.coe_inj]
    /-
      case neg.intro
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      f : 𝕜 → E
      x : 𝕜
      hf : MeromorphicAt f x
      n : Int
      h✝ : Not (Eq ⋯.order Top.top)
      m : Nat
      h : Eq ↑m ⋯.order
      ⊢ Iff (Eq (HSub.hSub ↑m ↑(Exists.choose hf)) n) (Exists fun g => And (Analytic …
    -/
    obtain ⟨g, hg_an, hg_ne, hg_eq⟩ := (AnalyticAt.order_eq_nat_iff _ _).mp h.symm
    replace hg_eq : ∀ᶠ (z : 𝕜) in 𝓝[≠] x, f z = (z - x) ^ (↑m - ↑hf.choose : ℤ) • g z := by
      rw [eventually_nhdsWithin_iff]
      filter_upwards [hg_eq] with z hg_eq hz
      rwa [← smul_right_inj <| zpow_ne_zero _ (sub_ne_zero.mpr hz), ← mul_smul,
        ← zpow_add₀ (sub_ne_zero.mpr hz), ← add_sub_assoc, add_sub_cancel_left, zpow_natCast,
        zpow_natCast]
    exact ⟨fun h ↦ ⟨g, hg_an, hg_ne, h ▸ hg_eq⟩,
      AnalyticAt.unique_eventuallyEq_zpow_smul_nonzero ⟨g, hg_an, hg_ne, hg_eq⟩⟩


/-- Compatibility of notions of `order` for analytic and meromorphic functions. -/
lemma _root_.AnalyticAt.meromorphicAt_order {f : 𝕜 → E} {x : 𝕜} (hf : AnalyticAt 𝕜 f x) :
    hf.meromorphicAt.order = hf.order.map (↑) := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : 𝕜 → E
    x : 𝕜
    hf : AnalyticAt 𝕜 f x
    ⊢ Eq ⋯.order (ENat.map Nat.cast hf.order)
  -/
  rcases eq_or_ne hf.order ⊤ with ho | ho
    /-
      case inl
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      f : 𝕜 → E
      x : 𝕜
      hf : AnalyticAt 𝕜 f x
      ho : Eq hf.order Top.top
      ⊢ Eq ⋯.order (ENat.map Nat.cast hf.order)
    -/
  · rw [ho, ENat.map_top, order_eq_top_iff]
    /-
      case inl
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      f : 𝕜 → E
      x : 𝕜
      hf : AnalyticAt 𝕜 f x
      ho : Eq hf.order Top.top
      ⊢ Filter.Eventually (fun z => Eq (f z) 0) (nhdsWithin x (HasCompl.compl (Singl …
    -/
    exact (hf.order_eq_top_iff.mp ho).filter_mono nhdsWithin_le_nhds
    /-
      🎉 no goals
    -/
    /-
      case inr
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      f : 𝕜 → E
      x : 𝕜
      hf : AnalyticAt 𝕜 f x
      ho : Ne hf.order Top.top
      ⊢ Eq ⋯.order (ENat.map Nat.cast hf.order)
    -/
  · obtain ⟨n, hn⟩ := ENat.ne_top_iff_exists.mp ho
    /-
      case inr.intro
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      f : 𝕜 → E
      x : 𝕜
      hf : AnalyticAt 𝕜 f x
      ho : Ne hf.order Top.top
      n : Nat
      hn : Eq (↑n) hf.order
      ⊢ Eq ⋯.order (ENat.map Nat.cast hf.order)
    -/
    simp_rw [← hn, ENat.map_coe, order_eq_int_iff, zpow_natCast]
    /-
      case inr.intro
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      f : 𝕜 → E
      x : 𝕜
      hf : AnalyticAt 𝕜 f x
      ho : Ne hf.order Top.top
      n : Nat
      hn : Eq (↑n) hf.order
      ⊢ Exists fun g => And (AnalyticAt 𝕜 g x) (And (Ne (g x) 0) (Filter.Eventually  …
    -/
    rcases (hf.order_eq_nat_iff _).mp hn.symm with ⟨g, h1, h2, h3⟩
    /-
      case inr.intro.intro.intro.intro
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      f : 𝕜 → E
      x : 𝕜
      hf : AnalyticAt 𝕜 f x
      ho : Ne hf.order Top.top
      n : Nat
      hn : Eq (↑n) hf.order
      g : 𝕜 → E
      h1 : AnalyticAt 𝕜 g x
      h2 : Ne (g x) 0
      h3 : Filter.Eventually (fun z => Eq (f z) (HSMul.hSMul (HPow.hPow (HSub.hSub z …
      ⊢ Exists fun g => And (AnalyticAt 𝕜 g x) (And (Ne (g x) 0) (Filter.Eventually  …
    -/
    exact ⟨g, h1, h2, h3.filter_mono nhdsWithin_le_nhds⟩
    /-
      🎉 no goals
    -/


lemma iff_eventuallyEq_zpow_smul_analyticAt {f : 𝕜 → E} {x : 𝕜} : MeromorphicAt f x ↔
    ∃ (n : ℤ) (g : 𝕜 → E), AnalyticAt 𝕜 g x ∧ ∀ᶠ z in 𝓝[≠] x, f z = (z - x) ^ n • g z := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : 𝕜 → E
    x : 𝕜
    ⊢ Iff (MeromorphicAt f x) (Exists fun n => Exists fun g => And (AnalyticAt 𝕜 g …
  -/
  refine ⟨fun ⟨n, hn⟩ ↦ ⟨-n, _, ⟨hn, eventually_nhdsWithin_iff.mpr ?_⟩⟩, ?_⟩
    /-
      case refine_1
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      f : 𝕜 → E
      x : 𝕜
      x✝ : MeromorphicAt f x
      n : Nat
      hn : AnalyticAt 𝕜 (fun z => HSMul.hSMul (HPow.hPow (HSub.hSub z x) n) (f z)) x
      ⊢ Filter.Eventually (fun x_1 => Membership.mem (HasCompl.compl (Singleton.sing …
    -/
  · filter_upwards with z hz
    /-
      case refine_1.h
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      f : 𝕜 → E
      x : 𝕜
      x✝ : MeromorphicAt f x
      n : Nat
      hn : AnalyticAt 𝕜 (fun z => HSMul.hSMul (HPow.hPow (HSub.hSub z x) n) (f z)) x
      z : 𝕜
      hz : Membership.mem (HasCompl.compl (Singleton.singleton x)) z
      ⊢ Eq (f z) (HSMul.hSMul (HPow.hPow (HSub.hSub z x) (Neg.neg ↑n)) (HSMul.hSMul  …
    -/
    match_scalars
    /-
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      f : 𝕜 → E
      x : 𝕜
      x✝ : MeromorphicAt f x
      n : Nat
      hn : AnalyticAt 𝕜 (fun z => HSMul.hSMul (HPow.hPow (HSub.hSub z x) n) (f z)) x
      z : 𝕜
      hz : Membership.mem (HasCompl.compl (Singleton.singleton x)) z
      ⊢ Eq 1 (HMul.hMul (HPow.hPow (HSub.hSub z x) (Neg.neg ↑n)) (HMul.hMul (HPow.hP …
    -/
    field_simp [sub_ne_zero.mpr hz]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      f : 𝕜 → E
      x : 𝕜
      ⊢ (Exists fun n => Exists fun g => And (AnalyticAt 𝕜 g x) (Filter.Eventually ( …
    -/
  · refine fun ⟨n, g, hg_an, hg_eq⟩ ↦ MeromorphicAt.congr ?_ (EventuallyEq.symm hg_eq)
    /-
      case refine_2
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      f : 𝕜 → E
      x : 𝕜
      x✝ : Exists fun n => Exists fun g => And (AnalyticAt 𝕜 g x) (Filter.Eventually …
      n : Int
      g : 𝕜 → E
      hg_an : AnalyticAt 𝕜 g x
      hg_eq : Filter.Eventually (fun z => Eq (f z) (HSMul.hSMul (HPow.hPow (HSub.hSu …
      ⊢ MeromorphicAt (fun x_1 => HSMul.hSMul (HPow.hPow (HSub.hSub x_1 x) n) (g x_1 …
    -/
    exact (((MeromorphicAt.id x).sub (.const _ x)).zpow _).smul hg_an.meromorphicAt
    /-
      🎉 no goals
    -/


/-- Meromorphy of a function on a set. -/
def MeromorphicOn (f : 𝕜 → E) (U : Set 𝕜) : Prop := ∀ x ∈ U, MeromorphicAt f x


lemma AnalyticOnNhd.meromorphicOn {f : 𝕜 → E} {U : Set 𝕜} (hf : AnalyticOnNhd 𝕜 f U) :
    MeromorphicOn f U :=
  fun x hx ↦ (hf x hx).meromorphicAt


@[deprecated (since := "2024-09-26")]
alias AnalyticOn.meromorphicOn := AnalyticOnNhd.meromorphicOn


lemma id {U : Set 𝕜} : MeromorphicOn id U := fun x _ ↦ .id x


lemma const (e : E) {U : Set 𝕜} : MeromorphicOn (fun _ ↦ e) U :=
  fun x _ ↦ .const e x


include hf in
lemma mono_set {V : Set 𝕜} (hv : V ⊆ U) : MeromorphicOn f V := fun x hx ↦ hf x (hv hx)


include hf hg in
lemma add : MeromorphicOn (f + g) U := fun x hx ↦ (hf x hx).add (hg x hx)


include hf hg in
lemma sub : MeromorphicOn (f - g) U := fun x hx ↦ (hf x hx).sub (hg x hx)


include hf in
lemma neg : MeromorphicOn (-f) U := fun x hx ↦ (hf x hx).neg


@[simp] lemma neg_iff : MeromorphicOn (-f) U ↔ MeromorphicOn f U :=
              /-
                𝕜 : Type u_1
                inst✝² : NontriviallyNormedField 𝕜
                E : Type u_2
                inst✝¹ : NormedAddCommGroup E
                inst✝ : NormedSpace 𝕜 E
                f : 𝕜 → E
                U : Set 𝕜
                h : MeromorphicOn (Neg.neg f) U
                ⊢ MeromorphicOn f U
              -/
  ⟨fun h ↦ by simpa only [neg_neg] using h.neg, neg⟩
              /-
                🎉 no goals
              -/


include hs hf in
lemma smul : MeromorphicOn (s • f) U := fun x hx ↦ (hs x hx).smul (hf x hx)


include hs ht in
lemma mul : MeromorphicOn (s * t) U := fun x hx ↦ (hs x hx).mul (ht x hx)


include hs in
lemma inv : MeromorphicOn s⁻¹ U := fun x hx ↦ (hs x hx).inv


@[simp] lemma inv_iff : MeromorphicOn s⁻¹ U ↔ MeromorphicOn s U :=
              /-
                𝕜 : Type u_1
                inst✝ : NontriviallyNormedField 𝕜
                s : 𝕜 → 𝕜
                U : Set 𝕜
                h : MeromorphicOn (Inv.inv s) U
                ⊢ MeromorphicOn s U
              -/
  ⟨fun h ↦ by simpa only [inv_inv] using h.inv, inv⟩
              /-
                🎉 no goals
              -/


include hs ht in
lemma div : MeromorphicOn (s / t) U := fun x hx ↦ (hs x hx).div (ht x hx)


include hs in
lemma pow (n : ℕ) : MeromorphicOn (s ^ n) U := fun x hx ↦ (hs x hx).pow _


include hs in
lemma zpow (n : ℤ) : MeromorphicOn (s ^ n) U := fun x hx ↦ (hs x hx).zpow _


include hf in
lemma congr (h_eq : Set.EqOn f g U) (hu : IsOpen U) : MeromorphicOn g U := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f g : 𝕜 → E
    U : Set 𝕜
    hf : MeromorphicOn f U
    h_eq : Set.EqOn f g U
    hu : IsOpen U
    ⊢ MeromorphicOn g U
  -/
  refine fun x hx ↦ (hf x hx).congr (EventuallyEq.filter_mono ?_ nhdsWithin_le_nhds)
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f g : 𝕜 → E
    U : Set 𝕜
    hf : MeromorphicOn f U
    h_eq : Set.EqOn f g U
    hu : IsOpen U
    x : 𝕜
    hx : Membership.mem U x
    ⊢ (nhds x).EventuallyEq f g
  -/
  exact eventually_of_mem (hu.mem_nhds hx) h_eq
  /-
    🎉 no goals
  -/


theorem eventually_codiscreteWithin_analyticAt
    [CompleteSpace E] (f : 𝕜 → E) (h : MeromorphicOn f U) :
    ∀ᶠ (y : 𝕜) in codiscreteWithin U, AnalyticAt 𝕜 f y := by
  /-
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    U : Set 𝕜
    inst✝ : CompleteSpace E
    f : 𝕜 → E
    h : MeromorphicOn f U
    ⊢ Filter.Eventually (fun y => AnalyticAt 𝕜 f y) (Filter.codiscreteWithin U)
  -/
  rw [eventually_iff, mem_codiscreteWithin]
  /-
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    U : Set 𝕜
    inst✝ : CompleteSpace E
    f : 𝕜 → E
    h : MeromorphicOn f U
    ⊢ ∀ (x : 𝕜), Membership.mem U x → Disjoint (nhdsWithin x (HasCompl.compl (Sing …
  -/
  intro x hx
  /-
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    U : Set 𝕜
    inst✝ : CompleteSpace E
    f : 𝕜 → E
    h : MeromorphicOn f U
    x : 𝕜
    hx : Membership.mem U x
    ⊢ Disjoint (nhdsWithin x (HasCompl.compl (Singleton.singleton x))) (Filter.pri …
  -/
  rw [disjoint_principal_right]
  /-
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    U : Set 𝕜
    inst✝ : CompleteSpace E
    f : 𝕜 → E
    h : MeromorphicOn f U
    x : 𝕜
    hx : Membership.mem U x
    ⊢ Membership.mem (nhdsWithin x (HasCompl.compl (Singleton.singleton x))) (HasC …
  -/
  apply Filter.mem_of_superset ((h x hx).eventually_analyticAt)
  /-
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    U : Set 𝕜
    inst✝ : CompleteSpace E
    f : 𝕜 → E
    h : MeromorphicOn f U
    x : 𝕜
    hx : Membership.mem U x
    ⊢ HasSubset.Subset (setOf fun x => (fun y => AnalyticAt 𝕜 f y) x) (HasCompl.co …
  -/
  intro x hx
  /-
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    U : Set 𝕜
    inst✝ : CompleteSpace E
    f : 𝕜 → E
    h : MeromorphicOn f U
    x✝ : 𝕜
    hx✝ : Membership.mem U x✝
    x : 𝕜
    hx : Membership.mem (setOf fun x => (fun y => AnalyticAt 𝕜 f y) x) x
    ⊢ Membership.mem (HasCompl.compl (SDiff.sdiff U (setOf fun x => AnalyticAt 𝕜 f …
  -/
  simp [hx]
  /-
    🎉 no goals
  -/


