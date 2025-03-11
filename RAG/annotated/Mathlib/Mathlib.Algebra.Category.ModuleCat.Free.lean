theorem disjoint_span_sum : Disjoint (span R (range (u ∘ Sum.inl)))
    (span R (range (u ∘ Sum.inr))) := by
  /-
    ι : Type u_1
    ι' : Type u_2
    R : Type u_3
    inst✝ : Ring R
    S : CategoryTheory.ShortComplex (ModuleCat R)
    hS : S.Exact
    v : ι → ↑S.X₁
    u : Sum ι ι' → ↑S.X₂
    hw : LinearIndependent R (Function.comp (⇑S.g.hom) (Function.comp u Sum.inr))
    huv : Eq (Function.comp u Sum.inl) (Function.comp (⇑S.f.hom) v)
    ⊢ Disjoint (Submodule.span R (Set.range (Function.comp u Sum.inl))) (Submodule …
  -/
  rw [huv, disjoint_comm]
  /-
    ι : Type u_1
    ι' : Type u_2
    R : Type u_3
    inst✝ : Ring R
    S : CategoryTheory.ShortComplex (ModuleCat R)
    hS : S.Exact
    v : ι → ↑S.X₁
    u : Sum ι ι' → ↑S.X₂
    hw : LinearIndependent R (Function.comp (⇑S.g.hom) (Function.comp u Sum.inr))
    huv : Eq (Function.comp u Sum.inl) (Function.comp (⇑S.f.hom) v)
    ⊢ Disjoint (Submodule.span R (Set.range (Function.comp u Sum.inr))) (Submodule …
  -/
  refine Disjoint.mono_right (span_mono (range_comp_subset_range _ _)) ?_
  /-
    ι : Type u_1
    ι' : Type u_2
    R : Type u_3
    inst✝ : Ring R
    S : CategoryTheory.ShortComplex (ModuleCat R)
    hS : S.Exact
    v : ι → ↑S.X₁
    u : Sum ι ι' → ↑S.X₂
    hw : LinearIndependent R (Function.comp (⇑S.g.hom) (Function.comp u Sum.inr))
    huv : Eq (Function.comp u Sum.inl) (Function.comp (⇑S.f.hom) v)
    ⊢ Disjoint (Submodule.span R (Set.range (Function.comp u Sum.inr))) (Submodule …
  -/
  rw [← LinearMap.range_coe, span_eq (LinearMap.range S.f.hom), hS.moduleCat_range_eq_ker]
  /-
    ι : Type u_1
    ι' : Type u_2
    R : Type u_3
    inst✝ : Ring R
    S : CategoryTheory.ShortComplex (ModuleCat R)
    hS : S.Exact
    v : ι → ↑S.X₁
    u : Sum ι ι' → ↑S.X₂
    hw : LinearIndependent R (Function.comp (⇑S.g.hom) (Function.comp u Sum.inr))
    huv : Eq (Function.comp u Sum.inl) (Function.comp (⇑S.f.hom) v)
    ⊢ Disjoint (Submodule.span R (Set.range (Function.comp u Sum.inr))) (LinearMap …
  -/
  exact range_ker_disjoint hw
  /-
    🎉 no goals
  -/


include hv hm in

/-- In the commutative diagram
```
             f     g
    0 --→ X₁ --→ X₂ --→ X₃
          ↑      ↑      ↑
         v|     u|     w|
          ι  → ι ⊕ ι' ← ι'
```
where the top row is an exact sequence of modules and the maps on the bottom are `Sum.inl` and
`Sum.inr`. If `u` is injective and `v` and `w` are linearly independent, then `u` is linearly
independent. -/
theorem linearIndependent_leftExact : LinearIndependent R u := by
  /-
    ι : Type u_1
    ι' : Type u_2
    R : Type u_3
    inst✝ : Ring R
    S : CategoryTheory.ShortComplex (ModuleCat R)
    hS : S.Exact
    v : ι → ↑S.X₁
    hv : LinearIndependent R v
    u : Sum ι ι' → ↑S.X₂
    hw : LinearIndependent R (Function.comp (⇑S.g.hom) (Function.comp u Sum.inr))
    hm : CategoryTheory.Mono S.f
    huv : Eq (Function.comp u Sum.inl) (Function.comp (⇑S.f.hom) v)
    ⊢ LinearIndependent R u
  -/
  rw [linearIndependent_sum]
  /-
    ι : Type u_1
    ι' : Type u_2
    R : Type u_3
    inst✝ : Ring R
    S : CategoryTheory.ShortComplex (ModuleCat R)
    hS : S.Exact
    v : ι → ↑S.X₁
    hv : LinearIndependent R v
    u : Sum ι ι' → ↑S.X₂
    hw : LinearIndependent R (Function.comp (⇑S.g.hom) (Function.comp u Sum.inr))
    hm : CategoryTheory.Mono S.f
    huv : Eq (Function.comp u Sum.inl) (Function.comp (⇑S.f.hom) v)
    ⊢ And (LinearIndependent R (Function.comp u Sum.inl)) (And (LinearIndependent  …
  -/
  refine ⟨?_, LinearIndependent.of_comp S.g.hom hw, disjoint_span_sum hS hw huv⟩
  /-
    ι : Type u_1
    ι' : Type u_2
    R : Type u_3
    inst✝ : Ring R
    S : CategoryTheory.ShortComplex (ModuleCat R)
    hS : S.Exact
    v : ι → ↑S.X₁
    hv : LinearIndependent R v
    u : Sum ι ι' → ↑S.X₂
    hw : LinearIndependent R (Function.comp (⇑S.g.hom) (Function.comp u Sum.inr))
    hm : CategoryTheory.Mono S.f
    huv : Eq (Function.comp u Sum.inl) (Function.comp (⇑S.f.hom) v)
    ⊢ LinearIndependent R (Function.comp u Sum.inl)
  -/
  rw [huv, LinearMap.linearIndependent_iff S.f.hom]; swap
    /-
      ι : Type u_1
      ι' : Type u_2
      R : Type u_3
      inst✝ : Ring R
      S : CategoryTheory.ShortComplex (ModuleCat R)
      hS : S.Exact
      v : ι → ↑S.X₁
      hv : LinearIndependent R v
      u : Sum ι ι' → ↑S.X₂
      hw : LinearIndependent R (Function.comp (⇑S.g.hom) (Function.comp u Sum.inr))
      hm : CategoryTheory.Mono S.f
      huv : Eq (Function.comp u Sum.inl) (Function.comp (⇑S.f.hom) v)
      ⊢ Eq (LinearMap.ker S.f.hom) Bot.bot
    -/
  · rw [LinearMap.ker_eq_bot, ← mono_iff_injective]
    /-
      ι : Type u_1
      ι' : Type u_2
      R : Type u_3
      inst✝ : Ring R
      S : CategoryTheory.ShortComplex (ModuleCat R)
      hS : S.Exact
      v : ι → ↑S.X₁
      hv : LinearIndependent R v
      u : Sum ι ι' → ↑S.X₂
      hw : LinearIndependent R (Function.comp (⇑S.g.hom) (Function.comp u Sum.inr))
      hm : CategoryTheory.Mono S.f
      huv : Eq (Function.comp u Sum.inl) (Function.comp (⇑S.f.hom) v)
      ⊢ CategoryTheory.Mono S.f
    -/
    infer_instance
    /-
      🎉 no goals
    -/
  /-
    ι : Type u_1
    ι' : Type u_2
    R : Type u_3
    inst✝ : Ring R
    S : CategoryTheory.ShortComplex (ModuleCat R)
    hS : S.Exact
    v : ι → ↑S.X₁
    hv : LinearIndependent R v
    u : Sum ι ι' → ↑S.X₂
    hw : LinearIndependent R (Function.comp (⇑S.g.hom) (Function.comp u Sum.inr))
    hm : CategoryTheory.Mono S.f
    huv : Eq (Function.comp u Sum.inl) (Function.comp (⇑S.f.hom) v)
    ⊢ LinearIndependent R v
  -/
  exact hv
  /-
    🎉 no goals
  -/


include hS' hv in
/-- Given a short exact sequence `0 ⟶ X₁ ⟶ X₂ ⟶ X₃ ⟶ 0` of `R`-modules and linearly independent
    families `v : ι → N` and `w : ι' → P`, we get a linearly independent family `ι ⊕ ι' → M` -/
theorem linearIndependent_shortExact {w : ι' → S.X₃} (hw : LinearIndependent R w) :
    LinearIndependent R (Sum.elim (S.f ∘ v) (S.g.hom.toFun.invFun ∘ w)) := by
  /-
    ι : Type u_1
    ι' : Type u_2
    R : Type u_3
    inst✝ : Ring R
    S : CategoryTheory.ShortComplex (ModuleCat R)
    hS' : S.ShortExact
    v : ι → ↑S.X₁
    hv : LinearIndependent R v
    w : ι' → ↑S.X₃
    hw : LinearIndependent R w
    ⊢ LinearIndependent R (Sum.elim (Function.comp (⇑S.f.hom) v) (Function.comp (F …
  -/
  apply linearIndependent_leftExact hS'.exact hv _ hS'.mono_f rfl
  /-
    ι : Type u_1
    ι' : Type u_2
    R : Type u_3
    inst✝ : Ring R
    S : CategoryTheory.ShortComplex (ModuleCat R)
    hS' : S.ShortExact
    v : ι → ↑S.X₁
    hv : LinearIndependent R v
    w : ι' → ↑S.X₃
    hw : LinearIndependent R w
    ⊢ LinearIndependent R (Function.comp (⇑S.g.hom) (Function.comp (Sum.elim (Func …
  -/
  dsimp
  /-
    ι : Type u_1
    ι' : Type u_2
    R : Type u_3
    inst✝ : Ring R
    S : CategoryTheory.ShortComplex (ModuleCat R)
    hS' : S.ShortExact
    v : ι → ↑S.X₁
    hv : LinearIndependent R v
    w : ι' → ↑S.X₃
    hw : LinearIndependent R w
    ⊢ LinearIndependent R (Function.comp (⇑S.g.hom) (Function.comp (Function.invFu …
  -/
  convert hw
  /-
    case h.e'_4
    ι : Type u_1
    ι' : Type u_2
    R : Type u_3
    inst✝ : Ring R
    S : CategoryTheory.ShortComplex (ModuleCat R)
    hS' : S.ShortExact
    v : ι → ↑S.X₁
    hv : LinearIndependent R v
    w : ι' → ↑S.X₃
    hw : LinearIndependent R w
    ⊢ Eq (Function.comp (⇑S.g.hom) (Function.comp (Function.invFun ⇑S.g.hom) w)) w
  -/
  ext
  /-
    case h.e'_4.h
    ι : Type u_1
    ι' : Type u_2
    R : Type u_3
    inst✝ : Ring R
    S : CategoryTheory.ShortComplex (ModuleCat R)
    hS' : S.ShortExact
    v : ι → ↑S.X₁
    hv : LinearIndependent R v
    w : ι' → ↑S.X₃
    hw : LinearIndependent R w
    x✝ : ι'
    ⊢ Eq (Function.comp (⇑S.g.hom) (Function.comp (Function.invFun ⇑S.g.hom) w) x✝ …
  -/
  apply Function.rightInverse_invFun ((epi_iff_surjective _).mp hS'.epi_g)
  /-
    🎉 no goals
  -/


include hS in
/-- In the commutative diagram
```
    f     g
 X₁ --→ X₂ --→ X₃
 ↑      ↑      ↑
v|     u|     w|
 ι  → ι ⊕ ι' ← ι'
```
where the top row is an exact sequence of modules and the maps on the bottom are `Sum.inl` and
`Sum.inr`. If `v` spans `X₁` and `w` spans `X₃`, then `u` spans `X₂`. -/
theorem span_exact {β : Type*} {u : ι ⊕ β → S.X₂} (huv : u ∘ Sum.inl = S.f ∘ v)
    (hv : ⊤ ≤ span R (range v))
    (hw : ⊤ ≤ span R (range (S.g ∘ u ∘ Sum.inr))) :
    ⊤ ≤ span R (range u) := by
  /-
    ι : Type u_1
    R : Type u_3
    inst✝ : Ring R
    S : CategoryTheory.ShortComplex (ModuleCat R)
    hS : S.Exact
    v : ι → ↑S.X₁
    β : Type u_4
    u : Sum ι β → ↑S.X₂
    huv : Eq (Function.comp u Sum.inl) (Function.comp (⇑S.f.hom) v)
    hv : LE.le Top.top (Submodule.span R (Set.range v))
    hw : LE.le Top.top (Submodule.span R (Set.range (Function.comp (⇑S.g.hom) (Fun …
    ⊢ LE.le Top.top (Submodule.span R (Set.range u))
  -/
  intro m _
  /-
    ι : Type u_1
    R : Type u_3
    inst✝ : Ring R
    S : CategoryTheory.ShortComplex (ModuleCat R)
    hS : S.Exact
    v : ι → ↑S.X₁
    β : Type u_4
    u : Sum ι β → ↑S.X₂
    huv : Eq (Function.comp u Sum.inl) (Function.comp (⇑S.f.hom) v)
    hv : LE.le Top.top (Submodule.span R (Set.range v))
    hw : LE.le Top.top (Submodule.span R (Set.range (Function.comp (⇑S.g.hom) (Fun …
    m : ↑S.X₂
    a✝ : Membership.mem Top.top m
    ⊢ Membership.mem (Submodule.span R (Set.range u)) m
  -/
  have hgm : S.g m ∈ span R (range (S.g ∘ u ∘ Sum.inr)) := hw mem_top
  /-
    ι : Type u_1
    R : Type u_3
    inst✝ : Ring R
    S : CategoryTheory.ShortComplex (ModuleCat R)
    hS : S.Exact
    v : ι → ↑S.X₁
    β : Type u_4
    u : Sum ι β → ↑S.X₂
    huv : Eq (Function.comp u Sum.inl) (Function.comp (⇑S.f.hom) v)
    hv : LE.le Top.top (Submodule.span R (Set.range v))
    hw : LE.le Top.top (Submodule.span R (Set.range (Function.comp (⇑S.g.hom) (Fun …
    m : ↑S.X₂
    a✝ : Membership.mem Top.top m
    hgm : Membership.mem (Submodule.span R (Set.range (Function.comp (⇑S.g.hom) (F …
    ⊢ Membership.mem (Submodule.span R (Set.range u)) m
  -/
  rw [Finsupp.mem_span_range_iff_exists_finsupp] at hgm
  /-
    ι : Type u_1
    R : Type u_3
    inst✝ : Ring R
    S : CategoryTheory.ShortComplex (ModuleCat R)
    hS : S.Exact
    v : ι → ↑S.X₁
    β : Type u_4
    u : Sum ι β → ↑S.X₂
    huv : Eq (Function.comp u Sum.inl) (Function.comp (⇑S.f.hom) v)
    hv : LE.le Top.top (Submodule.span R (Set.range v))
    hw : LE.le Top.top (Submodule.span R (Set.range (Function.comp (⇑S.g.hom) (Fun …
    m : ↑S.X₂
    a✝ : Membership.mem Top.top m
    hgm : Exists fun c => Eq (c.sum fun i a => HSMul.hSMul a (Function.comp (⇑S.g. …
    ⊢ Membership.mem (Submodule.span R (Set.range u)) m
  -/
  obtain ⟨cm, hm⟩ := hgm
  /-
    case intro
    ι : Type u_1
    R : Type u_3
    inst✝ : Ring R
    S : CategoryTheory.ShortComplex (ModuleCat R)
    hS : S.Exact
    v : ι → ↑S.X₁
    β : Type u_4
    u : Sum ι β → ↑S.X₂
    huv : Eq (Function.comp u Sum.inl) (Function.comp (⇑S.f.hom) v)
    hv : LE.le Top.top (Submodule.span R (Set.range v))
    hw : LE.le Top.top (Submodule.span R (Set.range (Function.comp (⇑S.g.hom) (Fun …
    m : ↑S.X₂
    a✝ : Membership.mem Top.top m
    cm : Finsupp β R
    hm : Eq (cm.sum fun i a => HSMul.hSMul a (Function.comp (⇑S.g.hom) (Function.c …
    ⊢ Membership.mem (Submodule.span R (Set.range u)) m
  -/
  let m' : S.X₂ := Finsupp.sum cm fun j a ↦ a • (u (Sum.inr j))
  have hsub : m - m' ∈ LinearMap.range S.f.hom := by
    rw [hS.moduleCat_range_eq_ker]
    simp only [LinearMap.mem_ker, map_sub, sub_eq_zero]
    rw [← hm, map_finsupp_sum]
    simp only [Function.comp_apply, map_smul]
  /-
    case intro
    ι : Type u_1
    R : Type u_3
    inst✝ : Ring R
    S : CategoryTheory.ShortComplex (ModuleCat R)
    hS : S.Exact
    v : ι → ↑S.X₁
    β : Type u_4
    u : Sum ι β → ↑S.X₂
    huv : Eq (Function.comp u Sum.inl) (Function.comp (⇑S.f.hom) v)
    hv : LE.le Top.top (Submodule.span R (Set.range v))
    hw : LE.le Top.top (Submodule.span R (Set.range (Function.comp (⇑S.g.hom) (Fun …
    m : ↑S.X₂
    a✝ : Membership.mem Top.top m
    cm : Finsupp β R
    hm : Eq (cm.sum fun i a => HSMul.hSMul a (Function.comp (⇑S.g.hom) (Function.c …
    m' : ↑S.X₂ := cm.sum fun j a => HSMul.hSMul a (u (Sum.inr j))
    hsub : Membership.mem (LinearMap.range S.f.hom) (HSub.hSub m m')
    ⊢ Membership.mem (Submodule.span R (Set.range u)) m
  -/
  obtain ⟨n, hnm⟩ := hsub
  /-
    case intro.intro
    ι : Type u_1
    R : Type u_3
    inst✝ : Ring R
    S : CategoryTheory.ShortComplex (ModuleCat R)
    hS : S.Exact
    v : ι → ↑S.X₁
    β : Type u_4
    u : Sum ι β → ↑S.X₂
    huv : Eq (Function.comp u Sum.inl) (Function.comp (⇑S.f.hom) v)
    hv : LE.le Top.top (Submodule.span R (Set.range v))
    hw : LE.le Top.top (Submodule.span R (Set.range (Function.comp (⇑S.g.hom) (Fun …
    m : ↑S.X₂
    a✝ : Membership.mem Top.top m
    cm : Finsupp β R
    hm : Eq (cm.sum fun i a => HSMul.hSMul a (Function.comp (⇑S.g.hom) (Function.c …
    m' : ↑S.X₂ := cm.sum fun j a => HSMul.hSMul a (u (Sum.inr j))
    n : ↑S.X₁
    hnm : Eq (S.f.hom n) (HSub.hSub m m')
    ⊢ Membership.mem (Submodule.span R (Set.range u)) m
  -/
  have hn : n ∈ span R (range v) := hv mem_top
  /-
    case intro.intro
    ι : Type u_1
    R : Type u_3
    inst✝ : Ring R
    S : CategoryTheory.ShortComplex (ModuleCat R)
    hS : S.Exact
    v : ι → ↑S.X₁
    β : Type u_4
    u : Sum ι β → ↑S.X₂
    huv : Eq (Function.comp u Sum.inl) (Function.comp (⇑S.f.hom) v)
    hv : LE.le Top.top (Submodule.span R (Set.range v))
    hw : LE.le Top.top (Submodule.span R (Set.range (Function.comp (⇑S.g.hom) (Fun …
    m : ↑S.X₂
    a✝ : Membership.mem Top.top m
    cm : Finsupp β R
    hm : Eq (cm.sum fun i a => HSMul.hSMul a (Function.comp (⇑S.g.hom) (Function.c …
    m' : ↑S.X₂ := cm.sum fun j a => HSMul.hSMul a (u (Sum.inr j))
    n : ↑S.X₁
    hnm : Eq (S.f.hom n) (HSub.hSub m m')
    hn : Membership.mem (Submodule.span R (Set.range v)) n
    ⊢ Membership.mem (Submodule.span R (Set.range u)) m
  -/
  rw [Finsupp.mem_span_range_iff_exists_finsupp] at hn
  /-
    case intro.intro
    ι : Type u_1
    R : Type u_3
    inst✝ : Ring R
    S : CategoryTheory.ShortComplex (ModuleCat R)
    hS : S.Exact
    v : ι → ↑S.X₁
    β : Type u_4
    u : Sum ι β → ↑S.X₂
    huv : Eq (Function.comp u Sum.inl) (Function.comp (⇑S.f.hom) v)
    hv : LE.le Top.top (Submodule.span R (Set.range v))
    hw : LE.le Top.top (Submodule.span R (Set.range (Function.comp (⇑S.g.hom) (Fun …
    m : ↑S.X₂
    a✝ : Membership.mem Top.top m
    cm : Finsupp β R
    hm : Eq (cm.sum fun i a => HSMul.hSMul a (Function.comp (⇑S.g.hom) (Function.c …
    m' : ↑S.X₂ := cm.sum fun j a => HSMul.hSMul a (u (Sum.inr j))
    n : ↑S.X₁
    hnm : Eq (S.f.hom n) (HSub.hSub m m')
    hn : Exists fun c => Eq (c.sum fun i a => HSMul.hSMul a (v i)) n
    ⊢ Membership.mem (Submodule.span R (Set.range u)) m
  -/
  obtain ⟨cn, hn⟩ := hn
  /-
    case intro.intro.intro
    ι : Type u_1
    R : Type u_3
    inst✝ : Ring R
    S : CategoryTheory.ShortComplex (ModuleCat R)
    hS : S.Exact
    v : ι → ↑S.X₁
    β : Type u_4
    u : Sum ι β → ↑S.X₂
    huv : Eq (Function.comp u Sum.inl) (Function.comp (⇑S.f.hom) v)
    hv : LE.le Top.top (Submodule.span R (Set.range v))
    hw : LE.le Top.top (Submodule.span R (Set.range (Function.comp (⇑S.g.hom) (Fun …
    m : ↑S.X₂
    a✝ : Membership.mem Top.top m
    cm : Finsupp β R
    hm : Eq (cm.sum fun i a => HSMul.hSMul a (Function.comp (⇑S.g.hom) (Function.c …
    m' : ↑S.X₂ := cm.sum fun j a => HSMul.hSMul a (u (Sum.inr j))
    n : ↑S.X₁
    hnm : Eq (S.f.hom n) (HSub.hSub m m')
    cn : Finsupp ι R
    hn : Eq (cn.sum fun i a => HSMul.hSMul a (v i)) n
    ⊢ Membership.mem (Submodule.span R (Set.range u)) m
  -/
  rw [← hn, map_finsupp_sum] at hnm
  /-
    case intro.intro.intro
    ι : Type u_1
    R : Type u_3
    inst✝ : Ring R
    S : CategoryTheory.ShortComplex (ModuleCat R)
    hS : S.Exact
    v : ι → ↑S.X₁
    β : Type u_4
    u : Sum ι β → ↑S.X₂
    huv : Eq (Function.comp u Sum.inl) (Function.comp (⇑S.f.hom) v)
    hv : LE.le Top.top (Submodule.span R (Set.range v))
    hw : LE.le Top.top (Submodule.span R (Set.range (Function.comp (⇑S.g.hom) (Fun …
    m : ↑S.X₂
    a✝ : Membership.mem Top.top m
    cm : Finsupp β R
    hm : Eq (cm.sum fun i a => HSMul.hSMul a (Function.comp (⇑S.g.hom) (Function.c …
    m' : ↑S.X₂ := cm.sum fun j a => HSMul.hSMul a (u (Sum.inr j))
    n : ↑S.X₁
    cn : Finsupp ι R
    hnm : Eq (cn.sum fun a b => S.f.hom (HSMul.hSMul b (v a))) (HSub.hSub m m')
    hn : Eq (cn.sum fun i a => HSMul.hSMul a (v i)) n
    ⊢ Membership.mem (Submodule.span R (Set.range u)) m
  -/
  rw [← sub_add_cancel m m', ← hnm,]
  /-
    case intro.intro.intro
    ι : Type u_1
    R : Type u_3
    inst✝ : Ring R
    S : CategoryTheory.ShortComplex (ModuleCat R)
    hS : S.Exact
    v : ι → ↑S.X₁
    β : Type u_4
    u : Sum ι β → ↑S.X₂
    huv : Eq (Function.comp u Sum.inl) (Function.comp (⇑S.f.hom) v)
    hv : LE.le Top.top (Submodule.span R (Set.range v))
    hw : LE.le Top.top (Submodule.span R (Set.range (Function.comp (⇑S.g.hom) (Fun …
    m : ↑S.X₂
    a✝ : Membership.mem Top.top m
    cm : Finsupp β R
    hm : Eq (cm.sum fun i a => HSMul.hSMul a (Function.comp (⇑S.g.hom) (Function.c …
    m' : ↑S.X₂ := cm.sum fun j a => HSMul.hSMul a (u (Sum.inr j))
    n : ↑S.X₁
    cn : Finsupp ι R
    hnm : Eq (cn.sum fun a b => S.f.hom (HSMul.hSMul b (v a))) (HSub.hSub m m')
    hn : Eq (cn.sum fun i a => HSMul.hSMul a (v i)) n
    ⊢ Membership.mem (Submodule.span R (Set.range u)) (HAdd.hAdd (cn.sum fun a b = …
  -/
  simp only [map_smul]
  have hn' : (Finsupp.sum cn fun a b ↦ b • S.f (v a)) =
      (Finsupp.sum cn fun a b ↦ b • u (Sum.inl a)) := by
    congr; ext a b; rw [← Function.comp_apply (f := S.f), ← huv, Function.comp_apply]
  /-
    case intro.intro.intro
    ι : Type u_1
    R : Type u_3
    inst✝ : Ring R
    S : CategoryTheory.ShortComplex (ModuleCat R)
    hS : S.Exact
    v : ι → ↑S.X₁
    β : Type u_4
    u : Sum ι β → ↑S.X₂
    huv : Eq (Function.comp u Sum.inl) (Function.comp (⇑S.f.hom) v)
    hv : LE.le Top.top (Submodule.span R (Set.range v))
    hw : LE.le Top.top (Submodule.span R (Set.range (Function.comp (⇑S.g.hom) (Fun …
    m : ↑S.X₂
    a✝ : Membership.mem Top.top m
    cm : Finsupp β R
    hm : Eq (cm.sum fun i a => HSMul.hSMul a (Function.comp (⇑S.g.hom) (Function.c …
    m' : ↑S.X₂ := cm.sum fun j a => HSMul.hSMul a (u (Sum.inr j))
    n : ↑S.X₁
    cn : Finsupp ι R
    hnm : Eq (cn.sum fun a b => S.f.hom (HSMul.hSMul b (v a))) (HSub.hSub m m')
    hn : Eq (cn.sum fun i a => HSMul.hSMul a (v i)) n
    hn' : Eq (cn.sum fun a b => HSMul.hSMul b (S.f.hom (v a))) (cn.sum fun a b =>  …
    ⊢ Membership.mem (Submodule.span R (Set.range u)) (HAdd.hAdd (cn.sum fun a b = …
  -/
  rw [hn']
  /-
    case intro.intro.intro
    ι : Type u_1
    R : Type u_3
    inst✝ : Ring R
    S : CategoryTheory.ShortComplex (ModuleCat R)
    hS : S.Exact
    v : ι → ↑S.X₁
    β : Type u_4
    u : Sum ι β → ↑S.X₂
    huv : Eq (Function.comp u Sum.inl) (Function.comp (⇑S.f.hom) v)
    hv : LE.le Top.top (Submodule.span R (Set.range v))
    hw : LE.le Top.top (Submodule.span R (Set.range (Function.comp (⇑S.g.hom) (Fun …
    m : ↑S.X₂
    a✝ : Membership.mem Top.top m
    cm : Finsupp β R
    hm : Eq (cm.sum fun i a => HSMul.hSMul a (Function.comp (⇑S.g.hom) (Function.c …
    m' : ↑S.X₂ := cm.sum fun j a => HSMul.hSMul a (u (Sum.inr j))
    n : ↑S.X₁
    cn : Finsupp ι R
    hnm : Eq (cn.sum fun a b => S.f.hom (HSMul.hSMul b (v a))) (HSub.hSub m m')
    hn : Eq (cn.sum fun i a => HSMul.hSMul a (v i)) n
    hn' : Eq (cn.sum fun a b => HSMul.hSMul b (S.f.hom (v a))) (cn.sum fun a b =>  …
    ⊢ Membership.mem (Submodule.span R (Set.range u)) (HAdd.hAdd (cn.sum fun a b = …
  -/
  apply add_mem
    /-
      case intro.intro.intro.a
      ι : Type u_1
      R : Type u_3
      inst✝ : Ring R
      S : CategoryTheory.ShortComplex (ModuleCat R)
      hS : S.Exact
      v : ι → ↑S.X₁
      β : Type u_4
      u : Sum ι β → ↑S.X₂
      huv : Eq (Function.comp u Sum.inl) (Function.comp (⇑S.f.hom) v)
      hv : LE.le Top.top (Submodule.span R (Set.range v))
      hw : LE.le Top.top (Submodule.span R (Set.range (Function.comp (⇑S.g.hom) (Fun …
      m : ↑S.X₂
      a✝ : Membership.mem Top.top m
      cm : Finsupp β R
      hm : Eq (cm.sum fun i a => HSMul.hSMul a (Function.comp (⇑S.g.hom) (Function.c …
      m' : ↑S.X₂ := cm.sum fun j a => HSMul.hSMul a (u (Sum.inr j))
      n : ↑S.X₁
      cn : Finsupp ι R
      hnm : Eq (cn.sum fun a b => S.f.hom (HSMul.hSMul b (v a))) (HSub.hSub m m')
      hn : Eq (cn.sum fun i a => HSMul.hSMul a (v i)) n
      hn' : Eq (cn.sum fun a b => HSMul.hSMul b (S.f.hom (v a))) (cn.sum fun a b =>  …
      ⊢ Membership.mem (Submodule.span R (Set.range u)) (cn.sum fun a b => HSMul.hSM …
    -/
  · rw [Finsupp.mem_span_range_iff_exists_finsupp]
    /-
      case intro.intro.intro.a
      ι : Type u_1
      R : Type u_3
      inst✝ : Ring R
      S : CategoryTheory.ShortComplex (ModuleCat R)
      hS : S.Exact
      v : ι → ↑S.X₁
      β : Type u_4
      u : Sum ι β → ↑S.X₂
      huv : Eq (Function.comp u Sum.inl) (Function.comp (⇑S.f.hom) v)
      hv : LE.le Top.top (Submodule.span R (Set.range v))
      hw : LE.le Top.top (Submodule.span R (Set.range (Function.comp (⇑S.g.hom) (Fun …
      m : ↑S.X₂
      a✝ : Membership.mem Top.top m
      cm : Finsupp β R
      hm : Eq (cm.sum fun i a => HSMul.hSMul a (Function.comp (⇑S.g.hom) (Function.c …
      m' : ↑S.X₂ := cm.sum fun j a => HSMul.hSMul a (u (Sum.inr j))
      n : ↑S.X₁
      cn : Finsupp ι R
      hnm : Eq (cn.sum fun a b => S.f.hom (HSMul.hSMul b (v a))) (HSub.hSub m m')
      hn : Eq (cn.sum fun i a => HSMul.hSMul a (v i)) n
      hn' : Eq (cn.sum fun a b => HSMul.hSMul b (S.f.hom (v a))) (cn.sum fun a b =>  …
      ⊢ Exists fun c => Eq (c.sum fun i a => HSMul.hSMul a (u i)) (cn.sum fun a b => …
    -/
    use cn.mapDomain (Sum.inl)
    /-
      case h
      ι : Type u_1
      R : Type u_3
      inst✝ : Ring R
      S : CategoryTheory.ShortComplex (ModuleCat R)
      hS : S.Exact
      v : ι → ↑S.X₁
      β : Type u_4
      u : Sum ι β → ↑S.X₂
      huv : Eq (Function.comp u Sum.inl) (Function.comp (⇑S.f.hom) v)
      hv : LE.le Top.top (Submodule.span R (Set.range v))
      hw : LE.le Top.top (Submodule.span R (Set.range (Function.comp (⇑S.g.hom) (Fun …
      m : ↑S.X₂
      a✝ : Membership.mem Top.top m
      cm : Finsupp β R
      hm : Eq (cm.sum fun i a => HSMul.hSMul a (Function.comp (⇑S.g.hom) (Function.c …
      m' : ↑S.X₂ := cm.sum fun j a => HSMul.hSMul a (u (Sum.inr j))
      n : ↑S.X₁
      cn : Finsupp ι R
      hnm : Eq (cn.sum fun a b => S.f.hom (HSMul.hSMul b (v a))) (HSub.hSub m m')
      hn : Eq (cn.sum fun i a => HSMul.hSMul a (v i)) n
      hn' : Eq (cn.sum fun a b => HSMul.hSMul b (S.f.hom (v a))) (cn.sum fun a b =>  …
      ⊢ Eq ((Finsupp.mapDomain Sum.inl cn).sum fun i a => HSMul.hSMul a (u i)) (cn.s …
    -/
    rw [Finsupp.sum_mapDomain_index_inj Sum.inl_injective]
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.a
      ι : Type u_1
      R : Type u_3
      inst✝ : Ring R
      S : CategoryTheory.ShortComplex (ModuleCat R)
      hS : S.Exact
      v : ι → ↑S.X₁
      β : Type u_4
      u : Sum ι β → ↑S.X₂
      huv : Eq (Function.comp u Sum.inl) (Function.comp (⇑S.f.hom) v)
      hv : LE.le Top.top (Submodule.span R (Set.range v))
      hw : LE.le Top.top (Submodule.span R (Set.range (Function.comp (⇑S.g.hom) (Fun …
      m : ↑S.X₂
      a✝ : Membership.mem Top.top m
      cm : Finsupp β R
      hm : Eq (cm.sum fun i a => HSMul.hSMul a (Function.comp (⇑S.g.hom) (Function.c …
      m' : ↑S.X₂ := cm.sum fun j a => HSMul.hSMul a (u (Sum.inr j))
      n : ↑S.X₁
      cn : Finsupp ι R
      hnm : Eq (cn.sum fun a b => S.f.hom (HSMul.hSMul b (v a))) (HSub.hSub m m')
      hn : Eq (cn.sum fun i a => HSMul.hSMul a (v i)) n
      hn' : Eq (cn.sum fun a b => HSMul.hSMul b (S.f.hom (v a))) (cn.sum fun a b =>  …
      ⊢ Membership.mem (Submodule.span R (Set.range u)) m'
    -/
  · rw [Finsupp.mem_span_range_iff_exists_finsupp]
    /-
      case intro.intro.intro.a
      ι : Type u_1
      R : Type u_3
      inst✝ : Ring R
      S : CategoryTheory.ShortComplex (ModuleCat R)
      hS : S.Exact
      v : ι → ↑S.X₁
      β : Type u_4
      u : Sum ι β → ↑S.X₂
      huv : Eq (Function.comp u Sum.inl) (Function.comp (⇑S.f.hom) v)
      hv : LE.le Top.top (Submodule.span R (Set.range v))
      hw : LE.le Top.top (Submodule.span R (Set.range (Function.comp (⇑S.g.hom) (Fun …
      m : ↑S.X₂
      a✝ : Membership.mem Top.top m
      cm : Finsupp β R
      hm : Eq (cm.sum fun i a => HSMul.hSMul a (Function.comp (⇑S.g.hom) (Function.c …
      m' : ↑S.X₂ := cm.sum fun j a => HSMul.hSMul a (u (Sum.inr j))
      n : ↑S.X₁
      cn : Finsupp ι R
      hnm : Eq (cn.sum fun a b => S.f.hom (HSMul.hSMul b (v a))) (HSub.hSub m m')
      hn : Eq (cn.sum fun i a => HSMul.hSMul a (v i)) n
      hn' : Eq (cn.sum fun a b => HSMul.hSMul b (S.f.hom (v a))) (cn.sum fun a b =>  …
      ⊢ Exists fun c => Eq (c.sum fun i a => HSMul.hSMul a (u i)) m'
    -/
    use cm.mapDomain (Sum.inr)
    /-
      case h
      ι : Type u_1
      R : Type u_3
      inst✝ : Ring R
      S : CategoryTheory.ShortComplex (ModuleCat R)
      hS : S.Exact
      v : ι → ↑S.X₁
      β : Type u_4
      u : Sum ι β → ↑S.X₂
      huv : Eq (Function.comp u Sum.inl) (Function.comp (⇑S.f.hom) v)
      hv : LE.le Top.top (Submodule.span R (Set.range v))
      hw : LE.le Top.top (Submodule.span R (Set.range (Function.comp (⇑S.g.hom) (Fun …
      m : ↑S.X₂
      a✝ : Membership.mem Top.top m
      cm : Finsupp β R
      hm : Eq (cm.sum fun i a => HSMul.hSMul a (Function.comp (⇑S.g.hom) (Function.c …
      m' : ↑S.X₂ := cm.sum fun j a => HSMul.hSMul a (u (Sum.inr j))
      n : ↑S.X₁
      cn : Finsupp ι R
      hnm : Eq (cn.sum fun a b => S.f.hom (HSMul.hSMul b (v a))) (HSub.hSub m m')
      hn : Eq (cn.sum fun i a => HSMul.hSMul a (v i)) n
      hn' : Eq (cn.sum fun a b => HSMul.hSMul b (S.f.hom (v a))) (cn.sum fun a b =>  …
      ⊢ Eq ((Finsupp.mapDomain Sum.inr cm).sum fun i a => HSMul.hSMul a (u i)) m'
    -/
    rw [Finsupp.sum_mapDomain_index_inj Sum.inr_injective]
    /-
      🎉 no goals
    -/


include hS in
/-- Given an exact sequence `X₁ ⟶ X₂ ⟶ X₃ ⟶ 0` of `R`-modules and spanning
    families `v : ι → X₁` and `w : ι' → X₃`, we get a spanning family `ι ⊕ ι' → X₂` -/
theorem span_rightExact {w : ι' → S.X₃} (hv : ⊤ ≤ span R (range v))
    (hw : ⊤ ≤ span R (range w)) (hE : Epi S.g) :
    ⊤ ≤ span R (range (Sum.elim (S.f ∘ v) (S.g.hom.toFun.invFun ∘ w))) := by
  /-
    ι : Type u_1
    ι' : Type u_2
    R : Type u_3
    inst✝ : Ring R
    S : CategoryTheory.ShortComplex (ModuleCat R)
    hS : S.Exact
    v : ι → ↑S.X₁
    w : ι' → ↑S.X₃
    hv : LE.le Top.top (Submodule.span R (Set.range v))
    hw : LE.le Top.top (Submodule.span R (Set.range w))
    hE : CategoryTheory.Epi S.g
    ⊢ LE.le Top.top (Submodule.span R (Set.range (Sum.elim (Function.comp (⇑S.f.ho …
  -/
  refine span_exact hS ?_ hv ?_
    /-
      case refine_1
      ι : Type u_1
      ι' : Type u_2
      R : Type u_3
      inst✝ : Ring R
      S : CategoryTheory.ShortComplex (ModuleCat R)
      hS : S.Exact
      v : ι → ↑S.X₁
      w : ι' → ↑S.X₃
      hv : LE.le Top.top (Submodule.span R (Set.range v))
      hw : LE.le Top.top (Submodule.span R (Set.range w))
      hE : CategoryTheory.Epi S.g
      ⊢ Eq (Function.comp (Sum.elim (Function.comp (⇑S.f.hom) v) (Function.comp (Fun …
    -/
  · simp only [AddHom.toFun_eq_coe, LinearMap.coe_toAddHom, Sum.elim_comp_inl]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      ι : Type u_1
      ι' : Type u_2
      R : Type u_3
      inst✝ : Ring R
      S : CategoryTheory.ShortComplex (ModuleCat R)
      hS : S.Exact
      v : ι → ↑S.X₁
      w : ι' → ↑S.X₃
      hv : LE.le Top.top (Submodule.span R (Set.range v))
      hw : LE.le Top.top (Submodule.span R (Set.range w))
      hE : CategoryTheory.Epi S.g
      ⊢ LE.le Top.top (Submodule.span R (Set.range (Function.comp (⇑S.g.hom) (Functi …
    -/
  · convert hw
    /-
      case h.e'_4.h.e'_6.h.e'_3
      ι : Type u_1
      ι' : Type u_2
      R : Type u_3
      inst✝ : Ring R
      S : CategoryTheory.ShortComplex (ModuleCat R)
      hS : S.Exact
      v : ι → ↑S.X₁
      w : ι' → ↑S.X₃
      hv : LE.le Top.top (Submodule.span R (Set.range v))
      hw : LE.le Top.top (Submodule.span R (Set.range w))
      hE : CategoryTheory.Epi S.g
      ⊢ Eq (Function.comp (⇑S.g.hom) (Function.comp (Sum.elim (Function.comp (⇑S.f.h …
    -/
    simp only [AddHom.toFun_eq_coe, LinearMap.coe_toAddHom, Sum.elim_comp_inr]
    /-
      case h.e'_4.h.e'_6.h.e'_3
      ι : Type u_1
      ι' : Type u_2
      R : Type u_3
      inst✝ : Ring R
      S : CategoryTheory.ShortComplex (ModuleCat R)
      hS : S.Exact
      v : ι → ↑S.X₁
      w : ι' → ↑S.X₃
      hv : LE.le Top.top (Submodule.span R (Set.range v))
      hw : LE.le Top.top (Submodule.span R (Set.range w))
      hE : CategoryTheory.Epi S.g
      ⊢ Eq (Function.comp (⇑S.g.hom) (Function.comp (Function.invFun ⇑S.g.hom) w)) w
    -/
    rw [ModuleCat.epi_iff_surjective] at hE
    rw [← Function.comp_assoc, Function.RightInverse.comp_eq_id (Function.rightInverse_invFun hE),
      Function.id_comp]


/-- In a short exact sequence `0 ⟶ X₁ ⟶ X₂ ⟶ X₃ ⟶ 0`, given bases for `X₁` and `X₃`
indexed by `ι` and `ι'` respectively, we get a basis for `X₂` indexed by `ι ⊕ ι'`. -/
noncomputable
def Basis.ofShortExact
    (bN : Basis ι R S.X₁) (bP : Basis ι' R S.X₃) : Basis (ι ⊕ ι') R S.X₂ :=
  Basis.mk (linearIndependent_shortExact hS' bN.linearIndependent bP.linearIndependent)
    (span_rightExact hS'.exact (le_of_eq (bN.span_eq.symm)) (le_of_eq (bP.span_eq.symm)) hS'.epi_g)


/-- In a short exact sequence `0 ⟶ X₁ ⟶ X₂ ⟶ X₃ ⟶ 0`, if `X₁` and `X₃` are free,
then `X₂` is free. -/
theorem free_shortExact [Module.Free R S.X₁] [Module.Free R S.X₃] :
    Module.Free R S.X₂ :=
  Module.Free.of_basis (Basis.ofShortExact hS' (Module.Free.chooseBasis R S.X₁)
    (Module.Free.chooseBasis R S.X₃))


theorem free_shortExact_rank_add [Module.Free R S.X₁] [Module.Free R S.X₃]
    [StrongRankCondition R] :
    Module.rank R S.X₂ = Module.rank R S.X₁ + Module.rank R S.X₃ := by
  /-
    R : Type u_3
    inst✝³ : Ring R
    S : CategoryTheory.ShortComplex (ModuleCat R)
    hS' : S.ShortExact
    inst✝² : Module.Free R ↑S.X₁
    inst✝¹ : Module.Free R ↑S.X₃
    inst✝ : StrongRankCondition R
    ⊢ Eq (Module.rank R ↑S.X₂) (HAdd.hAdd (Module.rank R ↑S.X₁) (Module.rank R ↑S. …
  -/
  haveI := free_shortExact hS'
  rw [Module.Free.rank_eq_card_chooseBasisIndex, Module.Free.rank_eq_card_chooseBasisIndex R S.X₁,
    Module.Free.rank_eq_card_chooseBasisIndex R S.X₃, Cardinal.add_def, Cardinal.eq]
  exact ⟨Basis.indexEquiv (Module.Free.chooseBasis R S.X₂) (Basis.ofShortExact hS'
    (Module.Free.chooseBasis R S.X₁) (Module.Free.chooseBasis R S.X₃))⟩


theorem free_shortExact_finrank_add {n p : ℕ} [Module.Free R S.X₁] [Module.Free R S.X₃]
    [Module.Finite R S.X₁] [Module.Finite R S.X₃]
    (hN : Module.finrank R S.X₁ = n)
    (hP : Module.finrank R S.X₃ = p)
    [StrongRankCondition R] :
    finrank R S.X₂ = n + p := by
  /-
    R : Type u_3
    inst✝⁵ : Ring R
    S : CategoryTheory.ShortComplex (ModuleCat R)
    hS' : S.ShortExact
    n p : Nat
    inst✝⁴ : Module.Free R ↑S.X₁
    inst✝³ : Module.Free R ↑S.X₃
    inst✝² : Module.Finite R ↑S.X₁
    inst✝¹ : Module.Finite R ↑S.X₃
    hN : Eq (Module.finrank R ↑S.X₁) n
    hP : Eq (Module.finrank R ↑S.X₃) p
    inst✝ : StrongRankCondition R
    ⊢ Eq (Module.finrank R ↑S.X₂) (HAdd.hAdd n p)
  -/
  apply finrank_eq_of_rank_eq
  /-
    case h
    R : Type u_3
    inst✝⁵ : Ring R
    S : CategoryTheory.ShortComplex (ModuleCat R)
    hS' : S.ShortExact
    n p : Nat
    inst✝⁴ : Module.Free R ↑S.X₁
    inst✝³ : Module.Free R ↑S.X₃
    inst✝² : Module.Finite R ↑S.X₁
    inst✝¹ : Module.Finite R ↑S.X₃
    hN : Eq (Module.finrank R ↑S.X₁) n
    hP : Eq (Module.finrank R ↑S.X₃) p
    inst✝ : StrongRankCondition R
    ⊢ Eq (Module.rank R ↑S.X₂) ↑(HAdd.hAdd n p)
  -/
  rw [free_shortExact_rank_add hS', ← hN, ← hP]
  /-
    case h
    R : Type u_3
    inst✝⁵ : Ring R
    S : CategoryTheory.ShortComplex (ModuleCat R)
    hS' : S.ShortExact
    n p : Nat
    inst✝⁴ : Module.Free R ↑S.X₁
    inst✝³ : Module.Free R ↑S.X₃
    inst✝² : Module.Finite R ↑S.X₁
    inst✝¹ : Module.Finite R ↑S.X₃
    hN : Eq (Module.finrank R ↑S.X₁) n
    hP : Eq (Module.finrank R ↑S.X₃) p
    inst✝ : StrongRankCondition R
    ⊢ Eq (HAdd.hAdd (Module.rank R ↑S.X₁) (Module.rank R ↑S.X₃)) ↑(HAdd.hAdd (Modu …
  -/
  simp only [Nat.cast_add, finrank_eq_rank]
  /-
    🎉 no goals
  -/


