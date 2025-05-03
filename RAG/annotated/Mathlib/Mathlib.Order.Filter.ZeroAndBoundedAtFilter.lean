/-- If `l` is a filter on `α`, then a function `f : α → β` is `ZeroAtFilter l`
  if it tends to zero along `l`. -/
def ZeroAtFilter [Zero β] [TopologicalSpace β] (l : Filter α) (f : α → β) : Prop :=
  Filter.Tendsto f l (𝓝 0)


theorem zero_zeroAtFilter [Zero β] [TopologicalSpace β] (l : Filter α) :
    ZeroAtFilter l (0 : α → β) :=
  tendsto_const_nhds


nonrec theorem ZeroAtFilter.add [TopologicalSpace β] [AddZeroClass β] [ContinuousAdd β]
    {l : Filter α} {f g : α → β} (hf : ZeroAtFilter l f) (hg : ZeroAtFilter l g) :
    ZeroAtFilter l (f + g) := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝² : TopologicalSpace β
    inst✝¹ : AddZeroClass β
    inst✝ : ContinuousAdd β
    l : Filter α
    f g : α → β
    hf : l.ZeroAtFilter f
    hg : l.ZeroAtFilter g
    ⊢ l.ZeroAtFilter (HAdd.hAdd f g)
  -/
  simpa using hf.add hg
  /-
    🎉 no goals
  -/


nonrec theorem ZeroAtFilter.neg [TopologicalSpace β] [AddGroup β] [ContinuousNeg β] {l : Filter α}
                                                                    /-
                                                                      α : Type u_2
                                                                      β : Type u_3
                                                                      inst✝² : TopologicalSpace β
                                                                      inst✝¹ : AddGroup β
                                                                      inst✝ : ContinuousNeg β
                                                                      l : Filter α
                                                                      f : α → β
                                                                      hf : l.ZeroAtFilter f
                                                                      ⊢ l.ZeroAtFilter (Neg.neg f)
                                                                    -/
    {f : α → β} (hf : ZeroAtFilter l f) : ZeroAtFilter l (-f) := by simpa using hf.neg
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


theorem ZeroAtFilter.smul [TopologicalSpace β] [Zero 𝕜] [Zero β]
    [SMulWithZero 𝕜 β] [ContinuousConstSMul 𝕜 β] {l : Filter α} {f : α → β} (c : 𝕜)
                                                           /-
                                                             𝕜 : Type u_1
                                                             α : Type u_2
                                                             β : Type u_3
                                                             inst✝⁴ : TopologicalSpace β
                                                             inst✝³ : Zero 𝕜
                                                             inst✝² : Zero β
                                                             inst✝¹ : SMulWithZero 𝕜 β
                                                             inst✝ : ContinuousConstSMul 𝕜 β
                                                             l : Filter α
                                                             f : α → β
                                                             c : 𝕜
                                                             hf : l.ZeroAtFilter f
                                                             ⊢ l.ZeroAtFilter (HSMul.hSMul c f)
                                                           -/
    (hf : ZeroAtFilter l f) : ZeroAtFilter l (c • f) := by simpa using hf.const_smul c
                                                           /-
                                                             🎉 no goals
                                                           -/


variable (𝕜) in
/-- `zeroAtFilterSubmodule l` is the submodule of `f : α → β` which
tend to zero along `l`. -/
def zeroAtFilterSubmodule
    [TopologicalSpace β] [Semiring 𝕜] [AddCommMonoid β] [Module 𝕜 β]
    [ContinuousAdd β] [ContinuousConstSMul 𝕜 β]
    (l : Filter α) : Submodule 𝕜 (α → β) where
  carrier := ZeroAtFilter l
  zero_mem' := zero_zeroAtFilter l
  add_mem' ha hb := ha.add hb
  smul_mem' c _ hf := hf.smul c


/-- `zeroAtFilterAddSubmonoid l` is the additive submonoid of `f : α → β`
which tend to zero along `l`. -/
def zeroAtFilterAddSubmonoid [TopologicalSpace β] [AddZeroClass β] [ContinuousAdd β]
    (l : Filter α) : AddSubmonoid (α → β) where
  carrier := ZeroAtFilter l
  add_mem' ha hb := ha.add hb
  zero_mem' := zero_zeroAtFilter l


/-- If `l` is a filter on `α`, then a function `f: α → β` is `BoundedAtFilter l`
if `f =O[l] 1`. -/
def BoundedAtFilter [Norm β] (l : Filter α) (f : α → β) : Prop :=
  Asymptotics.IsBigO l f (1 : α → ℝ)


theorem ZeroAtFilter.boundedAtFilter [SeminormedAddGroup β] {l : Filter α} {f : α → β}
    (hf : ZeroAtFilter l f) : BoundedAtFilter l f :=
  ((Asymptotics.isLittleO_one_iff _).mpr hf).isBigO


theorem const_boundedAtFilter [Norm β] (l : Filter α) (c : β) :
    BoundedAtFilter l (Function.const α c : α → β) :=
  Asymptotics.isBigO_const_const c one_ne_zero l

-- TODO(https://github.com/leanprover-community/mathlib4/issues/19288): Remove all Comm in the next
-- three lemmas. This would require modifying the corresponding general asymptotics lemma.

nonrec theorem BoundedAtFilter.add [SeminormedAddCommGroup β] {l : Filter α} {f g : α → β}
    (hf : BoundedAtFilter l f) (hg : BoundedAtFilter l g) : BoundedAtFilter l (f + g) := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝ : SeminormedAddCommGroup β
    l : Filter α
    f g : α → β
    hf : l.BoundedAtFilter f
    hg : l.BoundedAtFilter g
    ⊢ l.BoundedAtFilter (HAdd.hAdd f g)
  -/
  simpa using hf.add hg
  /-
    🎉 no goals
  -/


theorem BoundedAtFilter.neg [SeminormedAddCommGroup β] {l : Filter α} {f : α → β}
    (hf : BoundedAtFilter l f) : BoundedAtFilter l (-f) :=
  hf.neg_left


theorem BoundedAtFilter.smul
    [SeminormedRing 𝕜] [SeminormedAddCommGroup β] [Module 𝕜 β] [BoundedSMul 𝕜 β]
    {l : Filter α} {f : α → β} (c : 𝕜) (hf : BoundedAtFilter l f) : BoundedAtFilter l (c • f) :=
  hf.const_smul_left c


nonrec theorem BoundedAtFilter.mul [SeminormedRing β] {l : Filter α} {f g : α → β}
    (hf : BoundedAtFilter l f) (hg : BoundedAtFilter l g) : BoundedAtFilter l (f * g) := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝ : SeminormedRing β
    l : Filter α
    f g : α → β
    hf : l.BoundedAtFilter f
    hg : l.BoundedAtFilter g
    ⊢ l.BoundedAtFilter (HMul.hMul f g)
  -/
  refine (hf.mul hg).trans ?_
  /-
    α : Type u_2
    β : Type u_3
    inst✝ : SeminormedRing β
    l : Filter α
    f g : α → β
    hf : l.BoundedAtFilter f
    hg : l.BoundedAtFilter g
    ⊢ Asymptotics.IsBigO l (fun x => HMul.hMul (1 x) (1 x)) 1
  -/
  convert Asymptotics.isBigO_refl (E := ℝ) _ l
  /-
    case h.e'_8.h
    α : Type u_2
    β : Type u_3
    inst✝ : SeminormedRing β
    l : Filter α
    f g : α → β
    hf : l.BoundedAtFilter f
    hg : l.BoundedAtFilter g
    x✝ : α
    ⊢ Eq (1 x✝) (HMul.hMul (1 x✝) (1 x✝))
  -/
  simp
  /-
    🎉 no goals
  -/


variable (𝕜) in
/-- The submodule of functions that are bounded along a filter `l`. -/
def boundedFilterSubmodule
    [SeminormedRing 𝕜] [SeminormedAddCommGroup β] [Module 𝕜 β] [BoundedSMul 𝕜 β] (l : Filter α) :
    Submodule 𝕜 (α → β) where
  carrier := BoundedAtFilter l
  zero_mem' := const_boundedAtFilter l 0
  add_mem' hf hg := hf.add hg
  smul_mem' c _ hf := hf.smul c


variable (𝕜) in
/-- The subalgebra of functions that are bounded along a filter `l`. -/
def boundedFilterSubalgebra
    [SeminormedCommRing 𝕜] [SeminormedRing β] [Algebra 𝕜 β] [BoundedSMul 𝕜 β] (l : Filter α) :
    Subalgebra 𝕜 (α → β) :=
  Submodule.toSubalgebra
    (boundedFilterSubmodule 𝕜 l)
    (const_boundedAtFilter l (1 : β))
                        /-
                          𝕜 : Type u_1
                          α : Type u_2
                          β : Type u_3
                          inst✝³ : SeminormedCommRing 𝕜
                          inst✝² : SeminormedRing β
                          inst✝¹ : Algebra 𝕜 β
                          inst✝ : BoundedSMul 𝕜 β
                          l : Filter α
                          f g : α → β
                          hf : Membership.mem (Filter.boundedFilterSubmodule 𝕜 l) f
                          hg : Membership.mem (Filter.boundedFilterSubmodule 𝕜 l) g
                          ⊢ Membership.mem (Filter.boundedFilterSubmodule 𝕜 l) (HMul.hMul f g)
                        -/
    (fun f g hf hg ↦ by simpa only [Pi.one_apply, mul_one, norm_mul] using hf.mul hg)
                        /-
                          🎉 no goals
                        -/


