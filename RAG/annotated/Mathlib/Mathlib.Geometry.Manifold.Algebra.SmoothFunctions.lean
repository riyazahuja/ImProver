local notation "∞" => (⊤ : ℕ∞)


@[to_additive]
protected instance instMul {G : Type*} [Mul G] [TopologicalSpace G] [ChartedSpace H' G]
    [SmoothMul I' G] : Mul C^∞⟮I, N; I', G⟯ :=
  ⟨fun f g => ⟨f * g, f.contMDiff.mul g.contMDiff⟩⟩


@[to_additive (attr := simp)]
theorem coe_mul {G : Type*} [Mul G] [TopologicalSpace G] [ChartedSpace H' G] [SmoothMul I' G]
    (f g : C^∞⟮I, N; I', G⟯) : ⇑(f * g) = f * g :=
  rfl


@[to_additive (attr := simp)]
theorem mul_comp {G : Type*} [Mul G] [TopologicalSpace G] [ChartedSpace H' G] [SmoothMul I' G]
    (f g : C^∞⟮I'', N'; I', G⟯) (h : C^∞⟮I, N; I'', N'⟯) : (f * g).comp h = f.comp h * g.comp h :=
  rfl


@[to_additive]
protected instance instOne {G : Type*} [One G] [TopologicalSpace G] [ChartedSpace H' G] :
    One C^∞⟮I, N; I', G⟯ :=
  ⟨ContMDiffMap.const (1 : G)⟩


@[to_additive (attr := simp)]
theorem coe_one {G : Type*} [One G] [TopologicalSpace G] [ChartedSpace H' G] :
    ⇑(1 : C^∞⟮I, N; I', G⟯) = 1 :=
  rfl


instance instNSMul {G : Type*} [AddMonoid G] [TopologicalSpace G] [ChartedSpace H' G]
    [SmoothAdd I' G] : SMul ℕ C^∞⟮I, N; I', G⟯ where
  smul n f := ⟨n • (f : N → G), (contMDiff_nsmul n).comp f.contMDiff⟩


@[to_additive existing]
instance instPow {G : Type*} [Monoid G] [TopologicalSpace G] [ChartedSpace H' G] [SmoothMul I' G] :
    Pow C^∞⟮I, N; I', G⟯ ℕ where
  pow f n := ⟨(f : N → G) ^ n, (contMDiff_pow n).comp f.contMDiff⟩


@[to_additive (attr := simp)]
theorem coe_pow {G : Type*} [Monoid G] [TopologicalSpace G] [ChartedSpace H' G] [SmoothMul I' G]
    (f : C^∞⟮I, N; I', G⟯) (n : ℕ) :
    ⇑(f ^ n) = (f : N → G) ^ n :=
  rfl


@[to_additive]
instance semigroup {G : Type*} [Semigroup G] [TopologicalSpace G] [ChartedSpace H' G]
    [SmoothMul I' G] : Semigroup C^∞⟮I, N; I', G⟯ :=
  DFunLike.coe_injective.semigroup _ coe_mul


@[to_additive]
instance monoid {G : Type*} [Monoid G] [TopologicalSpace G] [ChartedSpace H' G]
    [SmoothMul I' G] : Monoid C^∞⟮I, N; I', G⟯ :=
  DFunLike.coe_injective.monoid _ coe_one coe_mul coe_pow


/-- Coercion to a function as a `MonoidHom`. Similar to `MonoidHom.coeFn`. -/
@[to_additive (attr := simps) "Coercion to a function as an `AddMonoidHom`.
  Similar to `AddMonoidHom.coeFn`."]
def coeFnMonoidHom {G : Type*} [Monoid G] [TopologicalSpace G] [ChartedSpace H' G]
    [SmoothMul I' G] : C^∞⟮I, N; I', G⟯ →* N → G where
  toFun := DFunLike.coe
  map_one' := coe_one
  map_mul' := coe_mul


/-- For a manifold `N` and a smooth homomorphism `φ` between Lie groups `G'`, `G''`, the
'left-composition-by-`φ`' group homomorphism from `C^∞⟮I, N; I', G'⟯` to `C^∞⟮I, N; I'', G''⟯`. -/
@[to_additive "For a manifold `N` and a smooth homomorphism `φ` between additive Lie groups `G'`,
`G''`, the 'left-composition-by-`φ`' group homomorphism from `C^∞⟮I, N; I', G'⟯` to
`C^∞⟮I, N; I'', G''⟯`."]
def compLeftMonoidHom {G' : Type*} [Monoid G'] [TopologicalSpace G'] [ChartedSpace H' G']
    [SmoothMul I' G'] {G'' : Type*} [Monoid G''] [TopologicalSpace G''] [ChartedSpace H'' G'']
    [SmoothMul I'' G''] (φ : G' →* G'') (hφ : ContMDiff I' I'' ⊤ φ) :
    C^∞⟮I, N; I', G'⟯ →* C^∞⟮I, N; I'', G''⟯ where
  toFun f := ⟨φ ∘ f, hφ.comp f.contMDiff⟩
                 /-
                   𝕜 : Type u_1
                   inst✝²¹ : NontriviallyNormedField 𝕜
                   E : Type u_2
                   inst✝²⁰ : NormedAddCommGroup E
                   inst✝¹⁹ : NormedSpace 𝕜 E
                   E' : Type u_3
                   inst✝¹⁸ : NormedAddCommGroup E'
                   inst✝¹⁷ : NormedSpace 𝕜 E'
                   H : Type u_4
                   inst✝¹⁶ : TopologicalSpace H
                   I : ModelWithCorners 𝕜 E H
                   H' : Type u_5
                   inst✝¹⁵ : TopologicalSpace H'
                   I' : ModelWithCorners 𝕜 E' H'
                   N : Type u_6
                   inst✝¹⁴ : TopologicalSpace N
                   inst✝¹³ : ChartedSpace H N
                   E'' : Type u_7
                   inst✝¹² : NormedAddCommGroup E''
                   inst✝¹¹ : NormedSpace 𝕜 E''
                   H'' : Type u_8
                   inst✝¹⁰ : TopologicalSpace H''
                   I'' : ModelWithCorners 𝕜 E'' H''
                   N' : Type u_9
                   inst✝⁹ : TopologicalSpace N'
                   inst✝⁸ : ChartedSpace H'' N'
                   G' : Type u_10
                   inst✝⁷ : Monoid G'
                   inst✝⁶ : TopologicalSpace G'
                   inst✝⁵ : ChartedSpace H' G'
                   inst✝⁴ : SmoothMul I' G'
                   G'' : Type u_11
                   inst✝³ : Monoid G''
                   inst✝² : TopologicalSpace G''
                   inst✝¹ : ChartedSpace H'' G''
                   inst✝ : SmoothMul I'' G''
                   φ : MonoidHom G' G''
                   hφ : ContMDiff I' I'' Top.top ⇑φ
                   ⊢ Eq ((fun f => ⟨Function.comp ⇑φ ⇑f, ⋯⟩) 1) 1
                 -/
  map_one' := by ext; show φ 1 = 1; simp
                                    /-
                                      🎉 no goals
                                    -/
                     /-
                       𝕜 : Type u_1
                       inst✝²¹ : NontriviallyNormedField 𝕜
                       E : Type u_2
                       inst✝²⁰ : NormedAddCommGroup E
                       inst✝¹⁹ : NormedSpace 𝕜 E
                       E' : Type u_3
                       inst✝¹⁸ : NormedAddCommGroup E'
                       inst✝¹⁷ : NormedSpace 𝕜 E'
                       H : Type u_4
                       inst✝¹⁶ : TopologicalSpace H
                       I : ModelWithCorners 𝕜 E H
                       H' : Type u_5
                       inst✝¹⁵ : TopologicalSpace H'
                       I' : ModelWithCorners 𝕜 E' H'
                       N : Type u_6
                       inst✝¹⁴ : TopologicalSpace N
                       inst✝¹³ : ChartedSpace H N
                       E'' : Type u_7
                       inst✝¹² : NormedAddCommGroup E''
                       inst✝¹¹ : NormedSpace 𝕜 E''
                       H'' : Type u_8
                       inst✝¹⁰ : TopologicalSpace H''
                       I'' : ModelWithCorners 𝕜 E'' H''
                       N' : Type u_9
                       inst✝⁹ : TopologicalSpace N'
                       inst✝⁸ : ChartedSpace H'' N'
                       G' : Type u_10
                       inst✝⁷ : Monoid G'
                       inst✝⁶ : TopologicalSpace G'
                       inst✝⁵ : ChartedSpace H' G'
                       inst✝⁴ : SmoothMul I' G'
                       G'' : Type u_11
                       inst✝³ : Monoid G''
                       inst✝² : TopologicalSpace G''
                       inst✝¹ : ChartedSpace H'' G''
                       inst✝ : SmoothMul I'' G''
                       φ : MonoidHom G' G''
                       hφ : ContMDiff I' I'' Top.top ⇑φ
                       f g : ContMDiffMap I I' N G' Top.top
                       ⊢ Eq ({ toFun := fun f => ⟨Function.comp ⇑φ ⇑f, ⋯⟩, map_one' := ⋯ }.toFun (HMu …
                     -/
  map_mul' f g := by ext x; show φ (f x * g x) = φ (f x) * φ (g x); simp
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


/-- For a Lie group `G` and open sets `U ⊆ V` in `N`, the 'restriction' group homomorphism from
`C^∞⟮I, V; I', G⟯` to `C^∞⟮I, U; I', G⟯`. -/
@[to_additive "For an additive Lie group `G` and open sets `U ⊆ V` in `N`, the 'restriction' group
homomorphism from `C^∞⟮I, V; I', G⟯` to `C^∞⟮I, U; I', G⟯`."]
def restrictMonoidHom (G : Type*) [Monoid G] [TopologicalSpace G] [ChartedSpace H' G]
    [SmoothMul I' G] {U V : Opens N} (h : U ≤ V) : C^∞⟮I, V; I', G⟯ →* C^∞⟮I, U; I', G⟯ where
  toFun f := ⟨f ∘ Set.inclusion h, f.contMDiff.comp (contMDiff_inclusion h)⟩
  map_one' := rfl
  map_mul' _ _ := rfl


@[to_additive]
instance commMonoid {G : Type*} [CommMonoid G] [TopologicalSpace G] [ChartedSpace H' G]
    [SmoothMul I' G] : CommMonoid C^∞⟮I, N; I', G⟯ :=
  DFunLike.coe_injective.commMonoid _ coe_one coe_mul coe_pow


@[to_additive]
instance group {G : Type*} [Group G] [TopologicalSpace G] [ChartedSpace H' G] [LieGroup I' G] :
    Group C^∞⟮I, N; I', G⟯ :=
  { SmoothMap.monoid with
    inv := fun f => ⟨fun x => (f x)⁻¹, f.contMDiff.inv⟩
                                  /-
                                    𝕜 : Type u_1
                                    inst✝¹⁷ : NontriviallyNormedField 𝕜
                                    E : Type u_2
                                    inst✝¹⁶ : NormedAddCommGroup E
                                    inst✝¹⁵ : NormedSpace 𝕜 E
                                    E' : Type u_3
                                    inst✝¹⁴ : NormedAddCommGroup E'
                                    inst✝¹³ : NormedSpace 𝕜 E'
                                    H : Type u_4
                                    inst✝¹² : TopologicalSpace H
                                    I : ModelWithCorners 𝕜 E H
                                    H' : Type u_5
                                    inst✝¹¹ : TopologicalSpace H'
                                    I' : ModelWithCorners 𝕜 E' H'
                                    N : Type u_6
                                    inst✝¹⁰ : TopologicalSpace N
                                    inst✝⁹ : ChartedSpace H N
                                    E'' : Type u_7
                                    inst✝⁸ : NormedAddCommGroup E''
                                    inst✝⁷ : NormedSpace 𝕜 E''
                                    H'' : Type u_8
                                    inst✝⁶ : TopologicalSpace H''
                                    I'' : ModelWithCorners 𝕜 E'' H''
                                    N' : Type u_9
                                    inst✝⁵ : TopologicalSpace N'
                                    inst✝⁴ : ChartedSpace H'' N'
                                    G : Type u_10
                                    inst✝³ : Group G
                                    inst✝² : TopologicalSpace G
                                    inst✝¹ : ChartedSpace H' G
                                    inst✝ : LieGroup I' G
                                    a : ContMDiffMap I I' N G Top.top
                                    ⊢ Eq (HMul.hMul (Inv.inv a) a) 1
                                  -/
    inv_mul_cancel := fun a => by ext; exact inv_mul_cancel _
                                    /-
                                      𝕜 : Type u_1
                                      inst✝¹⁷ : NontriviallyNormedField 𝕜
                                      E : Type u_2
                                      inst✝¹⁶ : NormedAddCommGroup E
                                      inst✝¹⁵ : NormedSpace 𝕜 E
                                      E' : Type u_3
                                      inst✝¹⁴ : NormedAddCommGroup E'
                                      inst✝¹³ : NormedSpace 𝕜 E'
                                      H : Type u_4
                                      inst✝¹² : TopologicalSpace H
                                      I : ModelWithCorners 𝕜 E H
                                      H' : Type u_5
                                      inst✝¹¹ : TopologicalSpace H'
                                      I' : ModelWithCorners 𝕜 E' H'
                                      N : Type u_6
                                      inst✝¹⁰ : TopologicalSpace N
                                      inst✝⁹ : ChartedSpace H N
                                      E'' : Type u_7
                                      inst✝⁸ : NormedAddCommGroup E''
                                      inst✝⁷ : NormedSpace 𝕜 E''
                                      H'' : Type u_8
                                      inst✝⁶ : TopologicalSpace H''
                                      I'' : ModelWithCorners 𝕜 E'' H''
                                      N' : Type u_9
                                      inst✝⁵ : TopologicalSpace N'
                                      inst✝⁴ : ChartedSpace H'' N'
                                      G : Type u_10
                                      inst✝³ : Group G
                                      inst✝² : TopologicalSpace G
                                      inst✝¹ : ChartedSpace H' G
                                      inst✝ : LieGroup I' G
                                      f g : ContMDiffMap I I' N G Top.top
                                      ⊢ Eq (HDiv.hDiv f g) (HMul.hMul f (Inv.inv g))
                                    -/
                                       /-
                                         🎉 no goals
                                       -/
                                         /-
                                           🎉 no goals
                                         -/
    div := fun f g => ⟨f / g, f.contMDiff.div g.contMDiff⟩
    div_eq_mul_inv := fun f g => by ext; exact div_eq_mul_inv _ _ }


@[to_additive (attr := simp)]
theorem coe_inv {G : Type*} [Group G] [TopologicalSpace G] [ChartedSpace H' G] [LieGroup I' G]
    (f : C^∞⟮I, N; I', G⟯) : ⇑f⁻¹ = (⇑f)⁻¹ :=
  rfl


@[to_additive (attr := simp)]
theorem coe_div {G : Type*} [Group G] [TopologicalSpace G] [ChartedSpace H' G] [LieGroup I' G]
    (f g : C^∞⟮I, N; I', G⟯) : ⇑(f / g) = f / g :=
  rfl


@[to_additive]
instance commGroup {G : Type*} [CommGroup G] [TopologicalSpace G] [ChartedSpace H' G]
    [LieGroup I' G] : CommGroup C^∞⟮I, N; I', G⟯ :=
  { SmoothMap.group, SmoothMap.commMonoid with }


instance semiring {R : Type*} [Semiring R] [TopologicalSpace R] [ChartedSpace H' R]
    [SmoothRing I' R] : Semiring C^∞⟮I, N; I', R⟯ :=
  { SmoothMap.addCommMonoid,
    SmoothMap.monoid with
                                    /-
                                      𝕜 : Type u_1
                                      inst✝¹⁷ : NontriviallyNormedField 𝕜
                                      E : Type u_2
                                      inst✝¹⁶ : NormedAddCommGroup E
                                      inst✝¹⁵ : NormedSpace 𝕜 E
                                      E' : Type u_3
                                      inst✝¹⁴ : NormedAddCommGroup E'
                                      inst✝¹³ : NormedSpace 𝕜 E'
                                      H : Type u_4
                                      inst✝¹² : TopologicalSpace H
                                      I : ModelWithCorners 𝕜 E H
                                      H' : Type u_5
                                      inst✝¹¹ : TopologicalSpace H'
                                      I' : ModelWithCorners 𝕜 E' H'
                                      N : Type u_6
                                      inst✝¹⁰ : TopologicalSpace N
                                      inst✝⁹ : ChartedSpace H N
                                      E'' : Type u_7
                                      inst✝⁸ : NormedAddCommGroup E''
                                      inst✝⁷ : NormedSpace 𝕜 E''
                                      H'' : Type u_8
                                      inst✝⁶ : TopologicalSpace H''
                                      I'' : ModelWithCorners 𝕜 E'' H''
                                      N' : Type u_9
                                      inst✝⁵ : TopologicalSpace N'
                                      inst✝⁴ : ChartedSpace H'' N'
                                      R : Type u_10
                                      inst✝³ : Semiring R
                                      inst✝² : TopologicalSpace R
                                      inst✝¹ : ChartedSpace H' R
                                      inst✝ : SmoothRing I' R
                                      a b c : ContMDiffMap I I' N R Top.top
                                      ⊢ Eq (HMul.hMul a (HAdd.hAdd b c)) (HAdd.hAdd (HMul.hMul a b) (HMul.hMul a c))
                                    -/
    left_distrib := fun a b c => by ext; exact left_distrib _ _ _
                                         /-
                                           🎉 no goals
                                         -/
                                     /-
                                       𝕜 : Type u_1
                                       inst✝¹⁷ : NontriviallyNormedField 𝕜
                                       E : Type u_2
                                       inst✝¹⁶ : NormedAddCommGroup E
                                       inst✝¹⁵ : NormedSpace 𝕜 E
                                       E' : Type u_3
                                       inst✝¹⁴ : NormedAddCommGroup E'
                                       inst✝¹³ : NormedSpace 𝕜 E'
                                       H : Type u_4
                                       inst✝¹² : TopologicalSpace H
                                       I : ModelWithCorners 𝕜 E H
                                       H' : Type u_5
                                       inst✝¹¹ : TopologicalSpace H'
                                       I' : ModelWithCorners 𝕜 E' H'
                                       N : Type u_6
                                       inst✝¹⁰ : TopologicalSpace N
                                       inst✝⁹ : ChartedSpace H N
                                       E'' : Type u_7
                                       inst✝⁸ : NormedAddCommGroup E''
                                       inst✝⁷ : NormedSpace 𝕜 E''
                                       H'' : Type u_8
                                       inst✝⁶ : TopologicalSpace H''
                                       I'' : ModelWithCorners 𝕜 E'' H''
                                       N' : Type u_9
                                       inst✝⁵ : TopologicalSpace N'
                                       inst✝⁴ : ChartedSpace H'' N'
                                       R : Type u_10
                                       inst✝³ : Semiring R
                                       inst✝² : TopologicalSpace R
                                       inst✝¹ : ChartedSpace H' R
                                       inst✝ : SmoothRing I' R
                                       a b c : ContMDiffMap I I' N R Top.top
                                       ⊢ Eq (HMul.hMul (HAdd.hAdd a b) c) (HAdd.hAdd (HMul.hMul a c) (HMul.hMul b c))
                                     -/
    right_distrib := fun a b c => by ext; exact right_distrib _ _ _
                                          /-
                                            🎉 no goals
                                          -/
                            /-
                              𝕜 : Type u_1
                              inst✝¹⁷ : NontriviallyNormedField 𝕜
                              E : Type u_2
                              inst✝¹⁶ : NormedAddCommGroup E
                              inst✝¹⁵ : NormedSpace 𝕜 E
                              E' : Type u_3
                              inst✝¹⁴ : NormedAddCommGroup E'
                              inst✝¹³ : NormedSpace 𝕜 E'
                              H : Type u_4
                              inst✝¹² : TopologicalSpace H
                              I : ModelWithCorners 𝕜 E H
                              H' : Type u_5
                              inst✝¹¹ : TopologicalSpace H'
                              I' : ModelWithCorners 𝕜 E' H'
                              N : Type u_6
                              inst✝¹⁰ : TopologicalSpace N
                              inst✝⁹ : ChartedSpace H N
                              E'' : Type u_7
                              inst✝⁸ : NormedAddCommGroup E''
                              inst✝⁷ : NormedSpace 𝕜 E''
                              H'' : Type u_8
                              inst✝⁶ : TopologicalSpace H''
                              I'' : ModelWithCorners 𝕜 E'' H''
                              N' : Type u_9
                              inst✝⁵ : TopologicalSpace N'
                              inst✝⁴ : ChartedSpace H'' N'
                              R : Type u_10
                              inst✝³ : Semiring R
                              inst✝² : TopologicalSpace R
                              inst✝¹ : ChartedSpace H' R
                              inst✝ : SmoothRing I' R
                              a : ContMDiffMap I I' N R Top.top
                              ⊢ Eq (HMul.hMul 0 a) 0
                            -/
    zero_mul := fun a => by ext; exact zero_mul _
                                 /-
                                   🎉 no goals
                                 -/
                            /-
                              𝕜 : Type u_1
                              inst✝¹⁷ : NontriviallyNormedField 𝕜
                              E : Type u_2
                              inst✝¹⁶ : NormedAddCommGroup E
                              inst✝¹⁵ : NormedSpace 𝕜 E
                              E' : Type u_3
                              inst✝¹⁴ : NormedAddCommGroup E'
                              inst✝¹³ : NormedSpace 𝕜 E'
                              H : Type u_4
                              inst✝¹² : TopologicalSpace H
                              I : ModelWithCorners 𝕜 E H
                              H' : Type u_5
                              inst✝¹¹ : TopologicalSpace H'
                              I' : ModelWithCorners 𝕜 E' H'
                              N : Type u_6
                              inst✝¹⁰ : TopologicalSpace N
                              inst✝⁹ : ChartedSpace H N
                              E'' : Type u_7
                              inst✝⁸ : NormedAddCommGroup E''
                              inst✝⁷ : NormedSpace 𝕜 E''
                              H'' : Type u_8
                              inst✝⁶ : TopologicalSpace H''
                              I'' : ModelWithCorners 𝕜 E'' H''
                              N' : Type u_9
                              inst✝⁵ : TopologicalSpace N'
                              inst✝⁴ : ChartedSpace H'' N'
                              R : Type u_10
                              inst✝³ : Semiring R
                              inst✝² : TopologicalSpace R
                              inst✝¹ : ChartedSpace H' R
                              inst✝ : SmoothRing I' R
                              a : ContMDiffMap I I' N R Top.top
                              ⊢ Eq (HMul.hMul a 0) 0
                            -/
    mul_zero := fun a => by ext; exact mul_zero _ }
                                 /-
                                   🎉 no goals
                                 -/


instance ring {R : Type*} [Ring R] [TopologicalSpace R] [ChartedSpace H' R] [SmoothRing I' R] :
    Ring C^∞⟮I, N; I', R⟯ :=
  { SmoothMap.semiring, SmoothMap.addCommGroup with }


instance commRing {R : Type*} [CommRing R] [TopologicalSpace R] [ChartedSpace H' R]
    [SmoothRing I' R] : CommRing C^∞⟮I, N; I', R⟯ :=
  { SmoothMap.semiring, SmoothMap.addCommGroup, SmoothMap.commMonoid with }


/-- For a manifold `N` and a smooth homomorphism `φ` between smooth rings `R'`, `R''`, the
'left-composition-by-`φ`' ring homomorphism from `C^∞⟮I, N; I', R'⟯` to `C^∞⟮I, N; I'', R''⟯`. -/
def compLeftRingHom {R' : Type*} [Ring R'] [TopologicalSpace R'] [ChartedSpace H' R']
    [SmoothRing I' R'] {R'' : Type*} [Ring R''] [TopologicalSpace R''] [ChartedSpace H'' R'']
    [SmoothRing I'' R''] (φ : R' →+* R'') (hφ : ContMDiff I' I'' ⊤ φ) :
    C^∞⟮I, N; I', R'⟯ →+* C^∞⟮I, N; I'', R''⟯ :=
  { SmoothMap.compLeftMonoidHom I N φ.toMonoidHom hφ,
    SmoothMap.compLeftAddMonoidHom I N φ.toAddMonoidHom hφ with
    toFun := fun f => ⟨φ ∘ f, hφ.comp f.contMDiff⟩ }


/-- For a "smooth ring" `R` and open sets `U ⊆ V` in `N`, the "restriction" ring homomorphism from
`C^∞⟮I, V; I', R⟯` to `C^∞⟮I, U; I', R⟯`. -/
def restrictRingHom (R : Type*) [Ring R] [TopologicalSpace R] [ChartedSpace H' R] [SmoothRing I' R]
    {U V : Opens N} (h : U ≤ V) : C^∞⟮I, V; I', R⟯ →+* C^∞⟮I, U; I', R⟯ :=
  { SmoothMap.restrictMonoidHom I I' R h, SmoothMap.restrictAddMonoidHom I I' R h with
    toFun := fun f => ⟨f ∘ Set.inclusion h, f.contMDiff.comp (contMDiff_inclusion h)⟩ }


/-- Coercion to a function as a `RingHom`. -/
@[simps]
def coeFnRingHom {R : Type*} [CommRing R] [TopologicalSpace R] [ChartedSpace H' R]
    [SmoothRing I' R] : C^∞⟮I, N; I', R⟯ →+* N → R :=
  { (coeFnMonoidHom : C^∞⟮I, N; I', R⟯ →* _), (coeFnAddMonoidHom : C^∞⟮I, N; I', R⟯ →+ _) with
    toFun := (↑) }


/-- `Function.eval` as a `RingHom` on the ring of smooth functions. -/
def evalRingHom {R : Type*} [CommRing R] [TopologicalSpace R] [ChartedSpace H' R] [SmoothRing I' R]
    (n : N) : C^∞⟮I, N; I', R⟯ →+* R :=
  (Pi.evalRingHom _ n : (N → R) →+* R).comp SmoothMap.coeFnRingHom


instance instSMul {V : Type*} [NormedAddCommGroup V] [NormedSpace 𝕜 V] :
    SMul 𝕜 C^∞⟮I, N; 𝓘(𝕜, V), V⟯ :=
  ⟨fun r f => ⟨r • ⇑f, contMDiff_const.smul f.contMDiff⟩⟩


@[simp]
theorem coe_smul {V : Type*} [NormedAddCommGroup V] [NormedSpace 𝕜 V] (r : 𝕜)
    (f : C^∞⟮I, N; 𝓘(𝕜, V), V⟯) : ⇑(r • f) = r • ⇑f :=
  rfl


@[simp]
theorem smul_comp {V : Type*} [NormedAddCommGroup V] [NormedSpace 𝕜 V] (r : 𝕜)
    (g : C^∞⟮I'', N'; 𝓘(𝕜, V), V⟯) (h : C^∞⟮I, N; I'', N'⟯) : (r • g).comp h = r • g.comp h :=
  rfl


instance module {V : Type*} [NormedAddCommGroup V] [NormedSpace 𝕜 V] :
    Module 𝕜 C^∞⟮I, N; 𝓘(𝕜, V), V⟯ :=
  Function.Injective.module 𝕜 coeFnAddMonoidHom ContMDiffMap.coe_injective coe_smul


/-- Coercion to a function as a `LinearMap`. -/
@[simps]
def coeFnLinearMap {V : Type*} [NormedAddCommGroup V] [NormedSpace 𝕜 V] :
    C^∞⟮I, N; 𝓘(𝕜, V), V⟯ →ₗ[𝕜] N → V :=
  { (coeFnAddMonoidHom : C^∞⟮I, N; 𝓘(𝕜, V), V⟯ →+ _) with
    toFun := (↑)
    map_smul' := coe_smul }


/-- Smooth constant functions as a `RingHom`. -/
def C : 𝕜 →+* C^∞⟮I, N; 𝓘(𝕜, A), A⟯ where
  toFun := fun c : 𝕜 => ⟨fun _ => (algebraMap 𝕜 A) c, contMDiff_const⟩
                 /-
                   𝕜 : Type u_1
                   inst✝¹⁶ : NontriviallyNormedField 𝕜
                   E : Type u_2
                   inst✝¹⁵ : NormedAddCommGroup E
                   inst✝¹⁴ : NormedSpace 𝕜 E
                   E' : Type u_3
                   inst✝¹³ : NormedAddCommGroup E'
                   inst✝¹² : NormedSpace 𝕜 E'
                   H : Type u_4
                   inst✝¹¹ : TopologicalSpace H
                   I : ModelWithCorners 𝕜 E H
                   H' : Type u_5
                   inst✝¹⁰ : TopologicalSpace H'
                   I' : ModelWithCorners 𝕜 E' H'
                   N : Type u_6
                   inst✝⁹ : TopologicalSpace N
                   inst✝⁸ : ChartedSpace H N
                   E'' : Type u_7
                   inst✝⁷ : NormedAddCommGroup E''
                   inst✝⁶ : NormedSpace 𝕜 E''
                   H'' : Type u_8
                   inst✝⁵ : TopologicalSpace H''
                   I'' : ModelWithCorners 𝕜 E'' H''
                   N' : Type u_9
                   inst✝⁴ : TopologicalSpace N'
                   inst✝³ : ChartedSpace H'' N'
                   A : Type u_10
                   inst✝² : NormedRing A
                   inst✝¹ : NormedAlgebra 𝕜 A
                   inst✝ : SmoothRing (modelWithCornersSelf 𝕜 A) A
                   ⊢ Eq ((fun c => ⟨fun x => (algebraMap 𝕜 A) c, ⋯⟩) 1) 1
                 -/
  map_one' := by ext; exact (algebraMap 𝕜 A).map_one
                      /-
                        🎉 no goals
                      -/
                       /-
                         𝕜 : Type u_1
                         inst✝¹⁶ : NontriviallyNormedField 𝕜
                         E : Type u_2
                         inst✝¹⁵ : NormedAddCommGroup E
                         inst✝¹⁴ : NormedSpace 𝕜 E
                         E' : Type u_3
                         inst✝¹³ : NormedAddCommGroup E'
                         inst✝¹² : NormedSpace 𝕜 E'
                         H : Type u_4
                         inst✝¹¹ : TopologicalSpace H
                         I : ModelWithCorners 𝕜 E H
                         H' : Type u_5
                         inst✝¹⁰ : TopologicalSpace H'
                         I' : ModelWithCorners 𝕜 E' H'
                         N : Type u_6
                         inst✝⁹ : TopologicalSpace N
                         inst✝⁸ : ChartedSpace H N
                         E'' : Type u_7
                         inst✝⁷ : NormedAddCommGroup E''
                         inst✝⁶ : NormedSpace 𝕜 E''
                         H'' : Type u_8
                         inst✝⁵ : TopologicalSpace H''
                         I'' : ModelWithCorners 𝕜 E'' H''
                         N' : Type u_9
                         inst✝⁴ : TopologicalSpace N'
                         inst✝³ : ChartedSpace H'' N'
                         A : Type u_10
                         inst✝² : NormedRing A
                         inst✝¹ : NormedAlgebra 𝕜 A
                         inst✝ : SmoothRing (modelWithCornersSelf 𝕜 A) A
                         c₁ c₂ : 𝕜
                         ⊢ Eq ({ toFun := fun c => ⟨fun x => (algebraMap 𝕜 A) c, ⋯⟩, map_one' := ⋯ }.to …
                       -/
  map_mul' c₁ c₂ := by ext; exact (algebraMap 𝕜 A).map_mul _ _
                            /-
                              🎉 no goals
                            -/
                  /-
                    𝕜 : Type u_1
                    inst✝¹⁶ : NontriviallyNormedField 𝕜
                    E : Type u_2
                    inst✝¹⁵ : NormedAddCommGroup E
                    inst✝¹⁴ : NormedSpace 𝕜 E
                    E' : Type u_3
                    inst✝¹³ : NormedAddCommGroup E'
                    inst✝¹² : NormedSpace 𝕜 E'
                    H : Type u_4
                    inst✝¹¹ : TopologicalSpace H
                    I : ModelWithCorners 𝕜 E H
                    H' : Type u_5
                    inst✝¹⁰ : TopologicalSpace H'
                    I' : ModelWithCorners 𝕜 E' H'
                    N : Type u_6
                    inst✝⁹ : TopologicalSpace N
                    inst✝⁸ : ChartedSpace H N
                    E'' : Type u_7
                    inst✝⁷ : NormedAddCommGroup E''
                    inst✝⁶ : NormedSpace 𝕜 E''
                    H'' : Type u_8
                    inst✝⁵ : TopologicalSpace H''
                    I'' : ModelWithCorners 𝕜 E'' H''
                    N' : Type u_9
                    inst✝⁴ : TopologicalSpace N'
                    inst✝³ : ChartedSpace H'' N'
                    A : Type u_10
                    inst✝² : NormedRing A
                    inst✝¹ : NormedAlgebra 𝕜 A
                    inst✝ : SmoothRing (modelWithCornersSelf 𝕜 A) A
                    ⊢ Eq ((↑{ toFun := fun c => ⟨fun x => (algebraMap 𝕜 A) c, ⋯⟩, map_one' := ⋯, m …
                  -/
  map_zero' := by ext; exact (algebraMap 𝕜 A).map_zero
                       /-
                         🎉 no goals
                       -/
                       /-
                         𝕜 : Type u_1
                         inst✝¹⁶ : NontriviallyNormedField 𝕜
                         E : Type u_2
                         inst✝¹⁵ : NormedAddCommGroup E
                         inst✝¹⁴ : NormedSpace 𝕜 E
                         E' : Type u_3
                         inst✝¹³ : NormedAddCommGroup E'
                         inst✝¹² : NormedSpace 𝕜 E'
                         H : Type u_4
                         inst✝¹¹ : TopologicalSpace H
                         I : ModelWithCorners 𝕜 E H
                         H' : Type u_5
                         inst✝¹⁰ : TopologicalSpace H'
                         I' : ModelWithCorners 𝕜 E' H'
                         N : Type u_6
                         inst✝⁹ : TopologicalSpace N
                         inst✝⁸ : ChartedSpace H N
                         E'' : Type u_7
                         inst✝⁷ : NormedAddCommGroup E''
                         inst✝⁶ : NormedSpace 𝕜 E''
                         H'' : Type u_8
                         inst✝⁵ : TopologicalSpace H''
                         I'' : ModelWithCorners 𝕜 E'' H''
                         N' : Type u_9
                         inst✝⁴ : TopologicalSpace N'
                         inst✝³ : ChartedSpace H'' N'
                         A : Type u_10
                         inst✝² : NormedRing A
                         inst✝¹ : NormedAlgebra 𝕜 A
                         inst✝ : SmoothRing (modelWithCornersSelf 𝕜 A) A
                         c₁ c₂ : 𝕜
                         ⊢ Eq ((↑{ toFun := fun c => ⟨fun x => (algebraMap 𝕜 A) c, ⋯⟩, map_one' := ⋯, m …
                       -/
  map_add' c₁ c₂ := by ext; exact (algebraMap 𝕜 A).map_add _ _
                            /-
                              🎉 no goals
                            -/


instance algebra : Algebra 𝕜 C^∞⟮I, N; 𝓘(𝕜, A), A⟯ :=
  { --SmoothMap.semiring with -- Porting note: Commented this out.
    smul := fun r f => ⟨r • f, contMDiff_const.smul f.contMDiff⟩
    toRingHom := SmoothMap.C
                               /-
                                 𝕜 : Type u_1
                                 inst✝¹⁶ : NontriviallyNormedField 𝕜
                                 E : Type u_2
                                 inst✝¹⁵ : NormedAddCommGroup E
                                 inst✝¹⁴ : NormedSpace 𝕜 E
                                 E' : Type u_3
                                 inst✝¹³ : NormedAddCommGroup E'
                                 inst✝¹² : NormedSpace 𝕜 E'
                                 H : Type u_4
                                 inst✝¹¹ : TopologicalSpace H
                                 I : ModelWithCorners 𝕜 E H
                                 H' : Type u_5
                                 inst✝¹⁰ : TopologicalSpace H'
                                 I' : ModelWithCorners 𝕜 E' H'
                                 N : Type u_6
                                 inst✝⁹ : TopologicalSpace N
                                 inst✝⁸ : ChartedSpace H N
                                 E'' : Type u_7
                                 inst✝⁷ : NormedAddCommGroup E''
                                 inst✝⁶ : NormedSpace 𝕜 E''
                                 H'' : Type u_8
                                 inst✝⁵ : TopologicalSpace H''
                                 I'' : ModelWithCorners 𝕜 E'' H''
                                 N' : Type u_9
                                 inst✝⁴ : TopologicalSpace N'
                                 inst✝³ : ChartedSpace H'' N'
                                 A : Type u_10
                                 inst✝² : NormedRing A
                                 inst✝¹ : NormedAlgebra 𝕜 A
                                 inst✝ : SmoothRing (modelWithCornersSelf 𝕜 A) A
                                 c : 𝕜
                                 f : ContMDiffMap I (modelWithCornersSelf 𝕜 A) N A Top.top
                                 ⊢ Eq (HMul.hMul (SmoothMap.C c) f) (HMul.hMul f (SmoothMap.C c))
                               -/
    commutes' := fun c f => by ext x; exact Algebra.commutes' _ _
                                      /-
                                        🎉 no goals
                                      -/
                               /-
                                 𝕜 : Type u_1
                                 inst✝¹⁶ : NontriviallyNormedField 𝕜
                                 E : Type u_2
                                 inst✝¹⁵ : NormedAddCommGroup E
                                 inst✝¹⁴ : NormedSpace 𝕜 E
                                 E' : Type u_3
                                 inst✝¹³ : NormedAddCommGroup E'
                                 inst✝¹² : NormedSpace 𝕜 E'
                                 H : Type u_4
                                 inst✝¹¹ : TopologicalSpace H
                                 I : ModelWithCorners 𝕜 E H
                                 H' : Type u_5
                                 inst✝¹⁰ : TopologicalSpace H'
                                 I' : ModelWithCorners 𝕜 E' H'
                                 N : Type u_6
                                 inst✝⁹ : TopologicalSpace N
                                 inst✝⁸ : ChartedSpace H N
                                 E'' : Type u_7
                                 inst✝⁷ : NormedAddCommGroup E''
                                 inst✝⁶ : NormedSpace 𝕜 E''
                                 H'' : Type u_8
                                 inst✝⁵ : TopologicalSpace H''
                                 I'' : ModelWithCorners 𝕜 E'' H''
                                 N' : Type u_9
                                 inst✝⁴ : TopologicalSpace N'
                                 inst✝³ : ChartedSpace H'' N'
                                 A : Type u_10
                                 inst✝² : NormedRing A
                                 inst✝¹ : NormedAlgebra 𝕜 A
                                 inst✝ : SmoothRing (modelWithCornersSelf 𝕜 A) A
                                 c : 𝕜
                                 f : ContMDiffMap I (modelWithCornersSelf 𝕜 A) N A Top.top
                                 ⊢ Eq (HSMul.hSMul c f) (HMul.hMul (SmoothMap.C c) f)
                               -/
    smul_def' := fun c f => by ext x; exact Algebra.smul_def' _ _ }
                                      /-
                                        🎉 no goals
                                      -/


/-- Coercion to a function as an `AlgHom`. -/
@[simps]
def coeFnAlgHom : C^∞⟮I, N; 𝓘(𝕜, A), A⟯ →ₐ[𝕜] N → A where
  toFun := (↑)
  commutes' _ := rfl
  -- `(SmoothMap.coeFnRingHom : C^∞⟮I, N; 𝓘(𝕜, A), A⟯ →+* _) with` times out for some reason
  map_zero' := SmoothMap.coe_zero
  map_one' := SmoothMap.coe_one
  map_add' := SmoothMap.coe_add
  map_mul' := SmoothMap.coe_mul


instance instSMul' {V : Type*} [NormedAddCommGroup V] [NormedSpace 𝕜 V] :
    SMul C^∞⟮I, N; 𝕜⟯ C^∞⟮I, N; 𝓘(𝕜, V), V⟯ :=
  ⟨fun f g => ⟨fun x => f x • g x, ContMDiff.smul f.2 g.2⟩⟩


@[simp]
theorem smul_comp' {V : Type*} [NormedAddCommGroup V] [NormedSpace 𝕜 V] (f : C^∞⟮I'', N'; 𝕜⟯)
    (g : C^∞⟮I'', N'; 𝓘(𝕜, V), V⟯) (h : C^∞⟮I, N; I'', N'⟯) :
    (f • g).comp h = f.comp h • g.comp h :=
  rfl


instance module' {V : Type*} [NormedAddCommGroup V] [NormedSpace 𝕜 V] :
    Module C^∞⟮I, N; 𝓘(𝕜), 𝕜⟯ C^∞⟮I, N; 𝓘(𝕜, V), V⟯ where
  smul := (· • ·)
                       /-
                         𝕜 : Type u_1
                         inst✝¹⁵ : NontriviallyNormedField 𝕜
                         E : Type u_2
                         inst✝¹⁴ : NormedAddCommGroup E
                         inst✝¹³ : NormedSpace 𝕜 E
                         E' : Type u_3
                         inst✝¹² : NormedAddCommGroup E'
                         inst✝¹¹ : NormedSpace 𝕜 E'
                         H : Type u_4
                         inst✝¹⁰ : TopologicalSpace H
                         I : ModelWithCorners 𝕜 E H
                         H' : Type u_5
                         inst✝⁹ : TopologicalSpace H'
                         I' : ModelWithCorners 𝕜 E' H'
                         N : Type u_6
                         inst✝⁸ : TopologicalSpace N
                         inst✝⁷ : ChartedSpace H N
                         E'' : Type u_7
                         inst✝⁶ : NormedAddCommGroup E''
                         inst✝⁵ : NormedSpace 𝕜 E''
                         H'' : Type u_8
                         inst✝⁴ : TopologicalSpace H''
                         I'' : ModelWithCorners 𝕜 E'' H''
                         N' : Type u_9
                         inst✝³ : TopologicalSpace N'
                         inst✝² : ChartedSpace H'' N'
                         V : Type u_10
                         inst✝¹ : NormedAddCommGroup V
                         inst✝ : NormedSpace 𝕜 V
                         c : ContMDiffMap I (modelWithCornersSelf 𝕜 𝕜) N 𝕜 Top.top
                         f g : ContMDiffMap I (modelWithCornersSelf 𝕜 V) N V Top.top
                         ⊢ Eq (HSMul.hSMul c (HAdd.hAdd f g)) (HAdd.hAdd (HSMul.hSMul c f) (HSMul.hSMul …
                       -/
  smul_add c f g := by ext x; exact smul_add (c x) (f x) (g x)
                         /-
                           𝕜 : Type u_1
                           inst✝¹⁵ : NontriviallyNormedField 𝕜
                           E : Type u_2
                           inst✝¹⁴ : NormedAddCommGroup E
                           inst✝¹³ : NormedSpace 𝕜 E
                           E' : Type u_3
                           inst✝¹² : NormedAddCommGroup E'
                           inst✝¹¹ : NormedSpace 𝕜 E'
                           H : Type u_4
                           inst✝¹⁰ : TopologicalSpace H
                           I : ModelWithCorners 𝕜 E H
                           H' : Type u_5
                           inst✝⁹ : TopologicalSpace H'
                           I' : ModelWithCorners 𝕜 E' H'
                           N : Type u_6
                           inst✝⁸ : TopologicalSpace N
                           inst✝⁷ : ChartedSpace H N
                           E'' : Type u_7
                           inst✝⁶ : NormedAddCommGroup E''
                           inst✝⁵ : NormedSpace 𝕜 E''
                           H'' : Type u_8
                           inst✝⁴ : TopologicalSpace H''
                           I'' : ModelWithCorners 𝕜 E'' H''
                           N' : Type u_9
                           inst✝³ : TopologicalSpace N'
                           inst✝² : ChartedSpace H'' N'
                           V : Type u_10
                           inst✝¹ : NormedAddCommGroup V
                           inst✝ : NormedSpace 𝕜 V
                           c₁ c₂ : ContMDiffMap I (modelWithCornersSelf 𝕜 𝕜) N 𝕜 Top.top
                           f : ContMDiffMap I (modelWithCornersSelf 𝕜 V) N V Top.top
                           ⊢ Eq (HSMul.hSMul (HMul.hMul c₁ c₂) f) (HSMul.hSMul c₁ (HSMul.hSMul c₂ f))
                         -/
                   /-
                     𝕜 : Type u_1
                     inst✝¹⁵ : NontriviallyNormedField 𝕜
                     E : Type u_2
                     inst✝¹⁴ : NormedAddCommGroup E
                     inst✝¹³ : NormedSpace 𝕜 E
                     E' : Type u_3
                     inst✝¹² : NormedAddCommGroup E'
                     inst✝¹¹ : NormedSpace 𝕜 E'
                     H : Type u_4
                     inst✝¹⁰ : TopologicalSpace H
                     I : ModelWithCorners 𝕜 E H
                     H' : Type u_5
                     inst✝⁹ : TopologicalSpace H'
                     I' : ModelWithCorners 𝕜 E' H'
                     N : Type u_6
                     inst✝⁸ : TopologicalSpace N
                     inst✝⁷ : ChartedSpace H N
                     E'' : Type u_7
                     inst✝⁶ : NormedAddCommGroup E''
                     inst✝⁵ : NormedSpace 𝕜 E''
                     H'' : Type u_8
                     inst✝⁴ : TopologicalSpace H''
                     I'' : ModelWithCorners 𝕜 E'' H''
                     N' : Type u_9
                     inst✝³ : TopologicalSpace N'
                     inst✝² : ChartedSpace H'' N'
                     V : Type u_10
                     inst✝¹ : NormedAddCommGroup V
                     inst✝ : NormedSpace 𝕜 V
                     f : ContMDiffMap I (modelWithCornersSelf 𝕜 V) N V Top.top
                     ⊢ Eq (HSMul.hSMul 1 f) f
                   -/
                              /-
                                🎉 no goals
                              -/
                          /-
                            🎉 no goals
                          -/
                                /-
                                  🎉 no goals
                                -/
                         /-
                           𝕜 : Type u_1
                           inst✝¹⁵ : NontriviallyNormedField 𝕜
                           E : Type u_2
                           inst✝¹⁴ : NormedAddCommGroup E
                           inst✝¹³ : NormedSpace 𝕜 E
                           E' : Type u_3
                           inst✝¹² : NormedAddCommGroup E'
                           inst✝¹¹ : NormedSpace 𝕜 E'
                           H : Type u_4
                           inst✝¹⁰ : TopologicalSpace H
                           I : ModelWithCorners 𝕜 E H
                           H' : Type u_5
                           inst✝⁹ : TopologicalSpace H'
                           I' : ModelWithCorners 𝕜 E' H'
                           N : Type u_6
                           inst✝⁸ : TopologicalSpace N
                           inst✝⁷ : ChartedSpace H N
                           E'' : Type u_7
                           inst✝⁶ : NormedAddCommGroup E''
                           inst✝⁵ : NormedSpace 𝕜 E''
                           H'' : Type u_8
                           inst✝⁴ : TopologicalSpace H''
                           I'' : ModelWithCorners 𝕜 E'' H''
                           N' : Type u_9
                           inst✝³ : TopologicalSpace N'
                           inst✝² : ChartedSpace H'' N'
                           V : Type u_10
                           inst✝¹ : NormedAddCommGroup V
                           inst✝ : NormedSpace 𝕜 V
                           c₁ c₂ : ContMDiffMap I (modelWithCornersSelf 𝕜 𝕜) N 𝕜 Top.top
                           f : ContMDiffMap I (modelWithCornersSelf 𝕜 V) N V Top.top
                           ⊢ Eq (HSMul.hSMul (HAdd.hAdd c₁ c₂) f) (HAdd.hAdd (HSMul.hSMul c₁ f) (HSMul.hS …
                         -/
  add_smul c₁ c₂ f := by ext x; exact add_smul (c₁ x) (c₂ x) (f x)
                    /-
                      𝕜 : Type u_1
                      inst✝¹⁵ : NontriviallyNormedField 𝕜
                      E : Type u_2
                      inst✝¹⁴ : NormedAddCommGroup E
                      inst✝¹³ : NormedSpace 𝕜 E
                      E' : Type u_3
                      inst✝¹² : NormedAddCommGroup E'
                      inst✝¹¹ : NormedSpace 𝕜 E'
                      H : Type u_4
                      inst✝¹⁰ : TopologicalSpace H
                      I : ModelWithCorners 𝕜 E H
                      H' : Type u_5
                      inst✝⁹ : TopologicalSpace H'
                      I' : ModelWithCorners 𝕜 E' H'
                      N : Type u_6
                      inst✝⁸ : TopologicalSpace N
                      inst✝⁷ : ChartedSpace H N
                      E'' : Type u_7
                      inst✝⁶ : NormedAddCommGroup E''
                      inst✝⁵ : NormedSpace 𝕜 E''
                      H'' : Type u_8
                      inst✝⁴ : TopologicalSpace H''
                      I'' : ModelWithCorners 𝕜 E'' H''
                      N' : Type u_9
                      inst✝³ : TopologicalSpace N'
                      inst✝² : ChartedSpace H'' N'
                      V : Type u_10
                      inst✝¹ : NormedAddCommGroup V
                      inst✝ : NormedSpace 𝕜 V
                      r : ContMDiffMap I (modelWithCornersSelf 𝕜 𝕜) N 𝕜 Top.top
                      ⊢ Eq (HSMul.hSMul r 0) 0
                    -/
                                /-
                                  🎉 no goals
                                -/
                           /-
                             🎉 no goals
                           -/
  mul_smul c₁ c₂ f := by ext x; exact mul_smul (c₁ x) (c₂ x) (f x)
  one_smul f := by ext x; exact one_smul 𝕜 (f x)
                    /-
                      𝕜 : Type u_1
                      inst✝¹⁵ : NontriviallyNormedField 𝕜
                      E : Type u_2
                      inst✝¹⁴ : NormedAddCommGroup E
                      inst✝¹³ : NormedSpace 𝕜 E
                      E' : Type u_3
                      inst✝¹² : NormedAddCommGroup E'
                      inst✝¹¹ : NormedSpace 𝕜 E'
                      H : Type u_4
                      inst✝¹⁰ : TopologicalSpace H
                      I : ModelWithCorners 𝕜 E H
                      H' : Type u_5
                      inst✝⁹ : TopologicalSpace H'
                      I' : ModelWithCorners 𝕜 E' H'
                      N : Type u_6
                      inst✝⁸ : TopologicalSpace N
                      inst✝⁷ : ChartedSpace H N
                      E'' : Type u_7
                      inst✝⁶ : NormedAddCommGroup E''
                      inst✝⁵ : NormedSpace 𝕜 E''
                      H'' : Type u_8
                      inst✝⁴ : TopologicalSpace H''
                      I'' : ModelWithCorners 𝕜 E'' H''
                      N' : Type u_9
                      inst✝³ : TopologicalSpace N'
                      inst✝² : ChartedSpace H'' N'
                      V : Type u_10
                      inst✝¹ : NormedAddCommGroup V
                      inst✝ : NormedSpace 𝕜 V
                      f : ContMDiffMap I (modelWithCornersSelf 𝕜 V) N V Top.top
                      ⊢ Eq (HSMul.hSMul 0 f) 0
                    -/
  zero_smul f := by ext x; exact zero_smul _ _
                           /-
                             🎉 no goals
                           -/
  smul_zero r := by ext x; exact smul_zero _


