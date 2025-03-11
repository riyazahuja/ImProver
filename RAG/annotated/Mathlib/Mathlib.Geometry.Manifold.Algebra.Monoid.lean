local notation "∞" => (⊤ : ℕ∞)


/-- Basic hypothesis to talk about a smooth (Lie) additive monoid or a smooth additive
semigroup. A smooth additive monoid over `α`, for example, is obtained by requiring both the
instances `AddMonoid α` and `SmoothAdd α`. -/
class SmoothAdd {𝕜 : Type*} [NontriviallyNormedField 𝕜] {H : Type*} [TopologicalSpace H]
    {E : Type*} [NormedAddCommGroup E] [NormedSpace 𝕜 E] (I : ModelWithCorners 𝕜 E H) (G : Type*)
    [Add G] [TopologicalSpace G] [ChartedSpace H G] extends SmoothManifoldWithCorners I G :
    Prop where
  smooth_add : ContMDiff (I.prod I) I ⊤ fun p : G × G => p.1 + p.2

-- See note [Design choices about smooth algebraic structures]

/-- Basic hypothesis to talk about a smooth (Lie) monoid or a smooth semigroup.
A smooth monoid over `G`, for example, is obtained by requiring both the instances `Monoid G`
and `SmoothMul I G`. -/
@[to_additive]
class SmoothMul {𝕜 : Type*} [NontriviallyNormedField 𝕜] {H : Type*} [TopologicalSpace H]
    {E : Type*} [NormedAddCommGroup E] [NormedSpace 𝕜 E] (I : ModelWithCorners 𝕜 E H) (G : Type*)
    [Mul G] [TopologicalSpace G] [ChartedSpace H G] extends SmoothManifoldWithCorners I G :
    Prop where
  smooth_mul : ContMDiff (I.prod I) I ⊤ fun p : G × G => p.1 * p.2


@[to_additive]
theorem contMDiff_mul : ContMDiff (I.prod I) I ⊤ fun p : G × G => p.1 * p.2 :=
  SmoothMul.smooth_mul


@[deprecated (since := "2024-11-20")] alias smooth_mul := contMDiff_mul

@[deprecated (since := "2024-11-20")] alias smooth_add := contMDiff_add


include I in
/-- If the multiplication is smooth, then it is continuous. This is not an instance for technical
reasons, see note [Design choices about smooth algebraic structures]. -/
@[to_additive "If the addition is smooth, then it is continuous. This is not an instance for
technical reasons, see note [Design choices about smooth algebraic structures]."]
theorem continuousMul_of_smooth : ContinuousMul G :=
  ⟨(contMDiff_mul I).continuous⟩


@[to_additive]
theorem ContMDiffWithinAt.mul (hf : ContMDiffWithinAt I' I n f s x)
    (hg : ContMDiffWithinAt I' I n g s x) : ContMDiffWithinAt I' I n (f * g) s x :=
  ((contMDiff_mul I).contMDiffAt.of_le le_top).comp_contMDiffWithinAt x (hf.prod_mk hg)


@[to_additive]
nonrec theorem ContMDiffAt.mul (hf : ContMDiffAt I' I n f x) (hg : ContMDiffAt I' I n g x) :
    ContMDiffAt I' I n (f * g) x :=
  hf.mul hg


@[to_additive]
theorem ContMDiffOn.mul (hf : ContMDiffOn I' I n f s) (hg : ContMDiffOn I' I n g s) :
    ContMDiffOn I' I n (f * g) s := fun x hx => (hf x hx).mul (hg x hx)


@[to_additive]
theorem ContMDiff.mul (hf : ContMDiff I' I n f) (hg : ContMDiff I' I n g) :
    ContMDiff I' I n (f * g) := fun x => (hf x).mul (hg x)


@[deprecated (since := "2024-11-21")] alias SmoothWithinAt.mul := ContMDiffWithinAt.mul

@[deprecated (since := "2024-11-21")] alias SmoothAt.mul := ContMDiffAt.mul

@[deprecated (since := "2024-11-21")] alias SmoothOn.mul := ContMDiffOn.mul

@[deprecated (since := "2024-11-21")] alias Smooth.mul := ContMDiff.mul


@[deprecated (since := "2024-11-21")] alias SmoothWithinAt.add := ContMDiffWithinAt.add

@[deprecated (since := "2024-11-21")] alias SmoothAt.add := ContMDiffAt.add

@[deprecated (since := "2024-11-21")] alias SmoothOn.add := ContMDiffOn.add

@[deprecated (since := "2024-11-21")] alias Smooth.add := ContMDiff.add


@[to_additive]
theorem contMDiff_mul_left {a : G} : ContMDiff I I n (a * ·) :=
  contMDiff_const.mul contMDiff_id


@[deprecated (since := "2024-11-21")] alias smooth_mul_left := contMDiff_mul_left

@[deprecated (since := "2024-11-21")] alias smooth_add_left := contMDiff_add_left


@[to_additive]
theorem contMDiffAt_mul_left {a b : G} : ContMDiffAt I I n (a * ·) b :=
  contMDiff_mul_left.contMDiffAt


@[to_additive]
theorem mdifferentiable_mul_left {a : G} : MDifferentiable I I (a * ·) :=
  contMDiff_mul_left.mdifferentiable le_rfl


@[to_additive]
theorem mdifferentiableAt_mul_left {a b : G} : MDifferentiableAt I I (a * ·) b :=
  contMDiffAt_mul_left.mdifferentiableAt le_rfl


@[to_additive]
theorem contMDiff_mul_right {a : G} : ContMDiff I I n (· * a) :=
  contMDiff_id.mul contMDiff_const


@[deprecated (since := "2024-11-21")] alias smooth_mul_right := contMDiff_mul_right

@[deprecated (since := "2024-11-21")] alias smooth_add_right := contMDiff_add_right


@[to_additive]
theorem contMDiffAt_mul_right {a b : G} : ContMDiffAt I I n (· * a) b :=
  contMDiff_mul_right.contMDiffAt


@[to_additive]
theorem mdifferentiable_mul_right {a : G} : MDifferentiable I I (· * a) :=
  contMDiff_mul_right.mdifferentiable le_rfl


@[to_additive]
theorem mdifferentiableAt_mul_right {a b : G} : MDifferentiableAt I I (· * a) b :=
  contMDiffAt_mul_right.mdifferentiableAt le_rfl


/-- Left multiplication by `g`. It is meant to mimic the usual notation in Lie groups.
Lemmas involving `smoothLeftMul` with the notation `𝑳` usually use `L` instead of `𝑳` in the
names. -/
def smoothLeftMul : C^∞⟮I, G; I, G⟯ :=
  ⟨leftMul g, contMDiff_mul_left⟩


/-- Right multiplication by `g`. It is meant to mimic the usual notation in Lie groups.
Lemmas involving `smoothRightMul` with the notation `𝑹` usually use `R` instead of `𝑹` in the
names. -/
def smoothRightMul : C^∞⟮I, G; I, G⟯ :=
  ⟨rightMul g, contMDiff_mul_right⟩

-- Left multiplication. The abbreviation is `MIL`.

@[inherit_doc] scoped[LieGroup] notation "𝑳" => smoothLeftMul

-- Right multiplication. The abbreviation is `MIR`.

@[inherit_doc] scoped[LieGroup] notation "𝑹" => smoothRightMul


@[simp]
theorem L_apply : (𝑳 I g) h = g * h :=
  rfl


@[simp]
theorem R_apply : (𝑹 I g) h = h * g :=
  rfl


@[simp]
theorem L_mul {G : Type*} [Semigroup G] [TopologicalSpace G] [ChartedSpace H G] [SmoothMul I G]
    (g h : G) : 𝑳 I (g * h) = (𝑳 I g).comp (𝑳 I h) := by
  /-
    𝕜 : Type u_1
    inst✝⁷ : NontriviallyNormedField 𝕜
    H : Type u_2
    inst✝⁶ : TopologicalSpace H
    E : Type u_3
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    I : ModelWithCorners 𝕜 E H
    G : Type u_8
    inst✝³ : Semigroup G
    inst✝² : TopologicalSpace G
    inst✝¹ : ChartedSpace H G
    inst✝ : SmoothMul I G
    g h : G
    ⊢ Eq (smoothLeftMul I (HMul.hMul g h)) ((smoothLeftMul I g).comp (smoothLeftMu …
  -/
  ext
  /-
    case h
    𝕜 : Type u_1
    inst✝⁷ : NontriviallyNormedField 𝕜
    H : Type u_2
    inst✝⁶ : TopologicalSpace H
    E : Type u_3
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    I : ModelWithCorners 𝕜 E H
    G : Type u_8
    inst✝³ : Semigroup G
    inst✝² : TopologicalSpace G
    inst✝¹ : ChartedSpace H G
    inst✝ : SmoothMul I G
    g h x✝ : G
    ⊢ Eq ((smoothLeftMul I (HMul.hMul g h)) x✝) (((smoothLeftMul I g).comp (smooth …
  -/
  simp only [ContMDiffMap.comp_apply, L_apply, mul_assoc]
  /-
    🎉 no goals
  -/


@[simp]
theorem R_mul {G : Type*} [Semigroup G] [TopologicalSpace G] [ChartedSpace H G] [SmoothMul I G]
    (g h : G) : 𝑹 I (g * h) = (𝑹 I h).comp (𝑹 I g) := by
  /-
    𝕜 : Type u_1
    inst✝⁷ : NontriviallyNormedField 𝕜
    H : Type u_2
    inst✝⁶ : TopologicalSpace H
    E : Type u_3
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    I : ModelWithCorners 𝕜 E H
    G : Type u_8
    inst✝³ : Semigroup G
    inst✝² : TopologicalSpace G
    inst✝¹ : ChartedSpace H G
    inst✝ : SmoothMul I G
    g h : G
    ⊢ Eq (smoothRightMul I (HMul.hMul g h)) ((smoothRightMul I h).comp (smoothRigh …
  -/
  ext
  /-
    case h
    𝕜 : Type u_1
    inst✝⁷ : NontriviallyNormedField 𝕜
    H : Type u_2
    inst✝⁶ : TopologicalSpace H
    E : Type u_3
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    I : ModelWithCorners 𝕜 E H
    G : Type u_8
    inst✝³ : Semigroup G
    inst✝² : TopologicalSpace G
    inst✝¹ : ChartedSpace H G
    inst✝ : SmoothMul I G
    g h x✝ : G
    ⊢ Eq ((smoothRightMul I (HMul.hMul g h)) x✝) (((smoothRightMul I h).comp (smoo …
  -/
  simp only [ContMDiffMap.comp_apply, R_apply, mul_assoc]
  /-
    🎉 no goals
  -/


theorem smoothLeftMul_one : (𝑳 I g') 1 = g' :=
  mul_one g'


theorem smoothRightMul_one : (𝑹 I g') 1 = g' :=
  one_mul g'


@[to_additive]
instance SmoothMul.prod {𝕜 : Type*} [NontriviallyNormedField 𝕜] {E : Type*}
    [NormedAddCommGroup E] [NormedSpace 𝕜 E] {H : Type*} [TopologicalSpace H]
    (I : ModelWithCorners 𝕜 E H) (G : Type*) [TopologicalSpace G] [ChartedSpace H G] [Mul G]
    [SmoothMul I G] {E' : Type*} [NormedAddCommGroup E'] [NormedSpace 𝕜 E'] {H' : Type*}
    [TopologicalSpace H'] (I' : ModelWithCorners 𝕜 E' H') (G' : Type*) [TopologicalSpace G']
    [ChartedSpace H' G'] [Mul G'] [SmoothMul I' G'] : SmoothMul (I.prod I') (G × G') :=
  { SmoothManifoldWithCorners.prod G G' with
    smooth_mul :=
      ((contMDiff_fst.comp contMDiff_fst).mul (contMDiff_fst.comp contMDiff_snd)).prod_mk
        ((contMDiff_snd.comp contMDiff_fst).mul (contMDiff_snd.comp contMDiff_snd)) }


@[to_additive]
theorem contMDiff_pow : ∀ n : ℕ, ContMDiff I I ⊤ fun a : G => a ^ n
            /-
              𝕜 : Type u_1
              inst✝⁷ : NontriviallyNormedField 𝕜
              H : Type u_2
              inst✝⁶ : TopologicalSpace H
              E : Type u_3
              inst✝⁵ : NormedAddCommGroup E
              inst✝⁴ : NormedSpace 𝕜 E
              I : ModelWithCorners 𝕜 E H
              G : Type u_4
              inst✝³ : Monoid G
              inst✝² : TopologicalSpace G
              inst✝¹ : ChartedSpace H G
              inst✝ : SmoothMul I G
              ⊢ ContMDiff I I Top.top fun a => HPow.hPow a 0
            -/
  | 0 => by simp only [pow_zero]; exact contMDiff_const
                                  /-
                                    🎉 no goals
                                  -/
                /-
                  𝕜 : Type u_1
                  inst✝⁷ : NontriviallyNormedField 𝕜
                  H : Type u_2
                  inst✝⁶ : TopologicalSpace H
                  E : Type u_3
                  inst✝⁵ : NormedAddCommGroup E
                  inst✝⁴ : NormedSpace 𝕜 E
                  I : ModelWithCorners 𝕜 E H
                  G : Type u_4
                  inst✝³ : Monoid G
                  inst✝² : TopologicalSpace G
                  inst✝¹ : ChartedSpace H G
                  inst✝ : SmoothMul I G
                  k : Nat
                  ⊢ ContMDiff I I Top.top fun a => HPow.hPow a (HAdd.hAdd k 1)
                -/
  | k + 1 => by simpa [pow_succ] using (contMDiff_pow _).mul contMDiff_id
                /-
                  🎉 no goals
                -/


@[deprecated (since := "2024-11-21")] alias smooth_pow := contMDiff_pow

@[deprecated (since := "2024-11-21")] alias smooth_nsmul := contMDiff_nsmul


/-- Morphism of additive smooth monoids. -/
structure SmoothAddMonoidMorphism (I : ModelWithCorners 𝕜 E H) (I' : ModelWithCorners 𝕜 E' H')
    (G : Type*) [TopologicalSpace G] [ChartedSpace H G] [AddMonoid G] [SmoothAdd I G]
    (G' : Type*) [TopologicalSpace G'] [ChartedSpace H' G'] [AddMonoid G']
    [SmoothAdd I' G'] extends G →+ G' where
  smooth_toFun : ContMDiff I I' ⊤ toFun


/-- Morphism of smooth monoids. -/
@[to_additive]
structure SmoothMonoidMorphism (I : ModelWithCorners 𝕜 E H) (I' : ModelWithCorners 𝕜 E' H')
    (G : Type*) [TopologicalSpace G] [ChartedSpace H G] [Monoid G] [SmoothMul I G] (G' : Type*)
    [TopologicalSpace G'] [ChartedSpace H' G'] [Monoid G'] [SmoothMul I' G'] extends
    G →* G' where
  smooth_toFun : ContMDiff I I' ⊤ toFun


@[to_additive]
instance : One (SmoothMonoidMorphism I I' G G') :=
  ⟨{  smooth_toFun := contMDiff_const
      toMonoidHom := 1 }⟩


@[to_additive]
instance : Inhabited (SmoothMonoidMorphism I I' G G') :=
  ⟨1⟩


@[to_additive]
instance : FunLike (SmoothMonoidMorphism I I' G G') G G' where
  coe a := a.toFun
                             /-
                               𝕜 : Type u_1
                               inst✝¹⁴ : NontriviallyNormedField 𝕜
                               H : Type u_2
                               inst✝¹³ : TopologicalSpace H
                               E : Type u_3
                               inst✝¹² : NormedAddCommGroup E
                               inst✝¹¹ : NormedSpace 𝕜 E
                               I : ModelWithCorners 𝕜 E H
                               G : Type u_4
                               inst✝¹⁰ : Monoid G
                               inst✝⁹ : TopologicalSpace G
                               inst✝⁸ : ChartedSpace H G
                               inst✝⁷ : SmoothMul I G
                               H' : Type u_5
                               inst✝⁶ : TopologicalSpace H'
                               E' : Type u_6
                               inst✝⁵ : NormedAddCommGroup E'
                               inst✝⁴ : NormedSpace 𝕜 E'
                               I' : ModelWithCorners 𝕜 E' H'
                               G' : Type u_7
                               inst✝³ : Monoid G'
                               inst✝² : TopologicalSpace G'
                               inst✝¹ : ChartedSpace H' G'
                               inst✝ : SmoothMul I' G'
                               f g : SmoothMonoidMorphism I I' G G'
                               h : Eq ((fun a => (↑a.toMonoidHom).toFun) f) ((fun a => (↑a.toMonoidHom).toFun …
                               ⊢ Eq f g
                             -/
  coe_injective' f g h := by cases f; cases g; congr; exact DFunLike.ext' h
                                                      /-
                                                        🎉 no goals
                                                      -/


@[to_additive]
instance : MonoidHomClass (SmoothMonoidMorphism I I' G G') G G' where
  map_one f := f.map_one
  map_mul f := f.map_mul


@[to_additive]
instance : ContinuousMapClass (SmoothMonoidMorphism I I' G G') G G' where
  map_continuous f := f.smooth_toFun.continuous


@[to_additive]
theorem ContMDiffWithinAt.prod (h : ∀ i ∈ t, ContMDiffWithinAt I' I n (f i) s x₀) :
    ContMDiffWithinAt I' I n (fun x ↦ ∏ i ∈ t, f i x) s x₀ := by
  classical
  induction' t using Finset.induction_on with i K iK IH
  · simp [contMDiffWithinAt_const]
  · simp only [iK, Finset.prod_insert, not_false_iff]
    exact (h _ (Finset.mem_insert_self i K)).mul (IH fun j hj ↦ h _ <| Finset.mem_insert_of_mem hj)


@[to_additive]
theorem contMDiffWithinAt_finprod (lf : LocallyFinite fun i ↦ mulSupport <| f i) {x₀ : M}
    (h : ∀ i, ContMDiffWithinAt I' I n (f i) s x₀) :
    ContMDiffWithinAt I' I n (fun x ↦ ∏ᶠ i, f i x) s x₀ :=
  let ⟨_I, hI⟩ := finprod_eventually_eq_prod lf x₀
  (ContMDiffWithinAt.prod fun i _hi ↦ h i).congr_of_eventuallyEq
    (eventually_nhdsWithin_of_eventually_nhds hI) hI.self_of_nhds


@[to_additive]
theorem contMDiffWithinAt_finset_prod' (h : ∀ i ∈ t, ContMDiffWithinAt I' I n (f i) s x) :
    ContMDiffWithinAt I' I n (∏ i ∈ t, f i) s x :=
  Finset.prod_induction f (fun f => ContMDiffWithinAt I' I n f s x) (fun _ _ hf hg => hf.mul hg)
    (contMDiffWithinAt_const (c := 1)) h


@[to_additive]
theorem contMDiffWithinAt_finset_prod (h : ∀ i ∈ t, ContMDiffWithinAt I' I n (f i) s x) :
    ContMDiffWithinAt I' I n (fun x => ∏ i ∈ t, f i x) s x := by
  /-
    ι : Type u_1
    𝕜 : Type u_2
    inst✝¹² : NontriviallyNormedField 𝕜
    H : Type u_3
    inst✝¹¹ : TopologicalSpace H
    E : Type u_4
    inst✝¹⁰ : NormedAddCommGroup E
    inst✝⁹ : NormedSpace 𝕜 E
    I : ModelWithCorners 𝕜 E H
    G : Type u_5
    inst✝⁸ : CommMonoid G
    inst✝⁷ : TopologicalSpace G
    inst✝⁶ : ChartedSpace H G
    inst✝⁵ : SmoothMul I G
    E' : Type u_6
    inst✝⁴ : NormedAddCommGroup E'
    inst✝³ : NormedSpace 𝕜 E'
    H' : Type u_7
    inst✝² : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M : Type u_8
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H' M
    s : Set M
    x : M
    t : Finset ι
    f : ι → M → G
    n : ENat
    h : ∀ (i : ι), Membership.mem t i → ContMDiffWithinAt I' I n (f i) s x
    ⊢ ContMDiffWithinAt I' I n (fun x => t.prod fun i => f i x) s x
  -/
  simp only [← Finset.prod_apply]
  /-
    ι : Type u_1
    𝕜 : Type u_2
    inst✝¹² : NontriviallyNormedField 𝕜
    H : Type u_3
    inst✝¹¹ : TopologicalSpace H
    E : Type u_4
    inst✝¹⁰ : NormedAddCommGroup E
    inst✝⁹ : NormedSpace 𝕜 E
    I : ModelWithCorners 𝕜 E H
    G : Type u_5
    inst✝⁸ : CommMonoid G
    inst✝⁷ : TopologicalSpace G
    inst✝⁶ : ChartedSpace H G
    inst✝⁵ : SmoothMul I G
    E' : Type u_6
    inst✝⁴ : NormedAddCommGroup E'
    inst✝³ : NormedSpace 𝕜 E'
    H' : Type u_7
    inst✝² : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M : Type u_8
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H' M
    s : Set M
    x : M
    t : Finset ι
    f : ι → M → G
    n : ENat
    h : ∀ (i : ι), Membership.mem t i → ContMDiffWithinAt I' I n (f i) s x
    ⊢ ContMDiffWithinAt I' I n (fun x => t.prod (fun c => f c) x) s x
  -/
  exact contMDiffWithinAt_finset_prod' h
  /-
    🎉 no goals
  -/


@[to_additive]
theorem ContMDiffAt.prod (h : ∀ i ∈ t, ContMDiffAt I' I n (f i) x₀) :
    ContMDiffAt I' I n (fun x ↦ ∏ i ∈ t, f i x) x₀ := by
  /-
    ι : Type u_1
    𝕜 : Type u_2
    inst✝¹² : NontriviallyNormedField 𝕜
    H : Type u_3
    inst✝¹¹ : TopologicalSpace H
    E : Type u_4
    inst✝¹⁰ : NormedAddCommGroup E
    inst✝⁹ : NormedSpace 𝕜 E
    I : ModelWithCorners 𝕜 E H
    G : Type u_5
    inst✝⁸ : CommMonoid G
    inst✝⁷ : TopologicalSpace G
    inst✝⁶ : ChartedSpace H G
    inst✝⁵ : SmoothMul I G
    E' : Type u_6
    inst✝⁴ : NormedAddCommGroup E'
    inst✝³ : NormedSpace 𝕜 E'
    H' : Type u_7
    inst✝² : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M : Type u_8
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H' M
    x₀ : M
    t : Finset ι
    f : ι → M → G
    n : ENat
    h : ∀ (i : ι), Membership.mem t i → ContMDiffAt I' I n (f i) x₀
    ⊢ ContMDiffAt I' I n (fun x => t.prod fun i => f i x) x₀
  -/
  simp only [← contMDiffWithinAt_univ] at *
  /-
    ι : Type u_1
    𝕜 : Type u_2
    inst✝¹² : NontriviallyNormedField 𝕜
    H : Type u_3
    inst✝¹¹ : TopologicalSpace H
    E : Type u_4
    inst✝¹⁰ : NormedAddCommGroup E
    inst✝⁹ : NormedSpace 𝕜 E
    I : ModelWithCorners 𝕜 E H
    G : Type u_5
    inst✝⁸ : CommMonoid G
    inst✝⁷ : TopologicalSpace G
    inst✝⁶ : ChartedSpace H G
    inst✝⁵ : SmoothMul I G
    E' : Type u_6
    inst✝⁴ : NormedAddCommGroup E'
    inst✝³ : NormedSpace 𝕜 E'
    H' : Type u_7
    inst✝² : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M : Type u_8
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H' M
    x₀ : M
    t : Finset ι
    f : ι → M → G
    n : ENat
    h : ∀ (i : ι), Membership.mem t i → ContMDiffWithinAt I' I n (f i) Set.univ x₀
    ⊢ ContMDiffWithinAt I' I n (fun x => t.prod fun i => f i x) Set.univ x₀
  -/
  exact ContMDiffWithinAt.prod h
  /-
    🎉 no goals
  -/


@[to_additive]
theorem contMDiffAt_finprod
    (lf : LocallyFinite fun i ↦ mulSupport <| f i) (h : ∀ i, ContMDiffAt I' I n (f i) x₀) :
    ContMDiffAt I' I n (fun x ↦ ∏ᶠ i, f i x) x₀ :=
  contMDiffWithinAt_finprod lf h


@[to_additive]
theorem contMDiffAt_finset_prod' (h : ∀ i ∈ t, ContMDiffAt I' I n (f i) x) :
    ContMDiffAt I' I n (∏ i ∈ t, f i) x :=
  contMDiffWithinAt_finset_prod' h


@[to_additive]
theorem contMDiffAt_finset_prod (h : ∀ i ∈ t, ContMDiffAt I' I n (f i) x) :
    ContMDiffAt I' I n (fun x => ∏ i ∈ t, f i x) x :=
  contMDiffWithinAt_finset_prod h


@[to_additive]
theorem contMDiffOn_finprod
    (lf : LocallyFinite fun i ↦ Function.mulSupport <| f i) (h : ∀ i, ContMDiffOn I' I n (f i) s) :
    ContMDiffOn I' I n (fun x ↦ ∏ᶠ i, f i x) s := fun x hx ↦
  contMDiffWithinAt_finprod lf fun i ↦ h i x hx


@[to_additive]
theorem contMDiffOn_finset_prod' (h : ∀ i ∈ t, ContMDiffOn I' I n (f i) s) :
    ContMDiffOn I' I n (∏ i ∈ t, f i) s := fun x hx =>
  contMDiffWithinAt_finset_prod' fun i hi => h i hi x hx


@[to_additive]
theorem contMDiffOn_finset_prod (h : ∀ i ∈ t, ContMDiffOn I' I n (f i) s) :
    ContMDiffOn I' I n (fun x => ∏ i ∈ t, f i x) s := fun x hx =>
  contMDiffWithinAt_finset_prod fun i hi => h i hi x hx


@[to_additive]
theorem ContMDiff.prod (h : ∀ i ∈ t, ContMDiff I' I n (f i)) :
    ContMDiff I' I n fun x ↦ ∏ i ∈ t, f i x :=
  fun x ↦ ContMDiffAt.prod fun j hj ↦ h j hj x


@[to_additive]
theorem contMDiff_finset_prod' (h : ∀ i ∈ t, ContMDiff I' I n (f i)) :
    ContMDiff I' I n (∏ i ∈ t, f i) := fun x => contMDiffAt_finset_prod' fun i hi => h i hi x


@[to_additive]
theorem contMDiff_finset_prod (h : ∀ i ∈ t, ContMDiff I' I n (f i)) :
    ContMDiff I' I n fun x => ∏ i ∈ t, f i x := fun x =>
  contMDiffAt_finset_prod fun i hi => h i hi x


@[to_additive]
theorem contMDiff_finprod (h : ∀ i, ContMDiff I' I n (f i))
    (hfin : LocallyFinite fun i => mulSupport (f i)) : ContMDiff I' I n fun x => ∏ᶠ i, f i x :=
  fun x ↦ contMDiffAt_finprod hfin fun i ↦ h i x


@[to_additive]
theorem contMDiff_finprod_cond (hc : ∀ i, p i → ContMDiff I' I n (f i))
    (hf : LocallyFinite fun i => mulSupport (f i)) :
    ContMDiff I' I n fun x => ∏ᶠ (i) (_ : p i), f i x := by
  /-
    ι : Type u_1
    𝕜 : Type u_2
    inst✝¹² : NontriviallyNormedField 𝕜
    H : Type u_3
    inst✝¹¹ : TopologicalSpace H
    E : Type u_4
    inst✝¹⁰ : NormedAddCommGroup E
    inst✝⁹ : NormedSpace 𝕜 E
    I : ModelWithCorners 𝕜 E H
    G : Type u_5
    inst✝⁸ : CommMonoid G
    inst✝⁷ : TopologicalSpace G
    inst✝⁶ : ChartedSpace H G
    inst✝⁵ : SmoothMul I G
    E' : Type u_6
    inst✝⁴ : NormedAddCommGroup E'
    inst✝³ : NormedSpace 𝕜 E'
    H' : Type u_7
    inst✝² : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M : Type u_8
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H' M
    f : ι → M → G
    n : ENat
    p : ι → Prop
    hc : ∀ (i : ι), p i → ContMDiff I' I n (f i)
    hf : LocallyFinite fun i => Function.mulSupport (f i)
    ⊢ ContMDiff I' I n fun x => finprod fun i => finprod fun x_1 => f i x
  -/
  simp only [← finprod_subtype_eq_finprod_cond]
  /-
    ι : Type u_1
    𝕜 : Type u_2
    inst✝¹² : NontriviallyNormedField 𝕜
    H : Type u_3
    inst✝¹¹ : TopologicalSpace H
    E : Type u_4
    inst✝¹⁰ : NormedAddCommGroup E
    inst✝⁹ : NormedSpace 𝕜 E
    I : ModelWithCorners 𝕜 E H
    G : Type u_5
    inst✝⁸ : CommMonoid G
    inst✝⁷ : TopologicalSpace G
    inst✝⁶ : ChartedSpace H G
    inst✝⁵ : SmoothMul I G
    E' : Type u_6
    inst✝⁴ : NormedAddCommGroup E'
    inst✝³ : NormedSpace 𝕜 E'
    H' : Type u_7
    inst✝² : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M : Type u_8
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H' M
    f : ι → M → G
    n : ENat
    p : ι → Prop
    hc : ∀ (i : ι), p i → ContMDiff I' I n (f i)
    hf : LocallyFinite fun i => Function.mulSupport (f i)
    ⊢ ContMDiff I' I n fun x => finprod fun j => f (↑j) x
  -/
  exact contMDiff_finprod (fun i => hc i i.2) (hf.comp_injective Subtype.coe_injective)
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-21")] alias smoothAt_finprod := contMDiffAt_finprod

@[deprecated (since := "2024-11-21")] alias smoothAt_finsum := contMDiffAt_finsum


@[deprecated (since := "2024-11-21")]
alias smoothWithinAt_finset_prod' := contMDiffWithinAt_finset_prod'

@[deprecated (since := "2024-11-21")]
alias smoothWithinAt_finset_sum' := contMDiffWithinAt_finset_sum'



@[deprecated (since := "2024-11-21")]
alias smoothWithinAt_finset_prod := contMDiffWithinAt_finset_prod

@[deprecated (since := "2024-11-21")]
alias smoothWithinAt_finset_sum := contMDiffWithinAt_finset_sum


@[deprecated (since := "2024-11-21")] alias smoothAt_finset_prod' := contMDiffAt_finset_prod'

@[deprecated (since := "2024-11-21")] alias smoothAt_finset_sum' := contMDiffAt_finset_sum'


@[deprecated (since := "2024-11-21")] alias smoothAt_finset_prod := contMDiffAt_finset_prod

@[deprecated (since := "2024-11-21")] alias smoothAt_finset_sum := contMDiffAt_finset_sum


@[deprecated (since := "2024-11-21")] alias smoothOn_finset_prod' := contMDiffOn_finset_prod'

@[deprecated (since := "2024-11-21")] alias smoothOn_finset_sum' := contMDiffOn_finset_sum'


@[deprecated (since := "2024-11-21")] alias smoothOn_finset_prod := contMDiffOn_finset_prod

@[deprecated (since := "2024-11-21")] alias smoothOn_finset_sum := contMDiffOn_finset_sum


@[deprecated (since := "2024-11-21")] alias smooth_finset_prod' := contMDiffOn_finset_prod'

@[deprecated (since := "2024-11-21")] alias smooth_finset_sum' := contMDiffOn_finset_sum'


@[deprecated (since := "2024-11-21")] alias smooth_finset_prod := contMDiff_finset_prod

@[deprecated (since := "2024-11-21")] alias smooth_finset_sum := contMDiff_finset_sum


@[deprecated (since := "2024-11-21")] alias smooth_finprod := contMDiff_finprod

@[deprecated (since := "2024-11-21")] alias smooth_finsum := contMDiff_finsum


@[deprecated (since := "2024-11-21")] alias smooth_finprod_cond := contMDiff_finprod_cond

@[deprecated (since := "2024-11-21")] alias smooth_finsum_cond := contMDiff_finsum_cond


instance hasSmoothAddSelf : SmoothAdd 𝓘(𝕜, E) E := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    ⊢ SmoothAdd (modelWithCornersSelf 𝕜 E) E
  -/
  constructor
  /-
    case smooth_add
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    ⊢ ContMDiff ((modelWithCornersSelf 𝕜 E).prod (modelWithCornersSelf 𝕜 E)) (mode …
  -/
  rw [← modelWithCornersSelf_prod, chartedSpaceSelf_prod]
  /-
    case smooth_add
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    ⊢ ContMDiff (modelWithCornersSelf 𝕜 (Prod E E)) (modelWithCornersSelf 𝕜 E) Top …
  -/
  exact contDiff_add.contMDiff
  /-
    🎉 no goals
  -/


@[to_additive]
theorem ContMDiffWithinAt.div_const (hf : ContMDiffWithinAt I' I n f s x) :
    ContMDiffWithinAt I' I n (fun x ↦ f x / c) s x := by
  /-
    𝕜 : Type u_1
    inst✝¹² : NontriviallyNormedField 𝕜
    H : Type u_2
    inst✝¹¹ : TopologicalSpace H
    E : Type u_3
    inst✝¹⁰ : NormedAddCommGroup E
    inst✝⁹ : NormedSpace 𝕜 E
    I : ModelWithCorners 𝕜 E H
    G : Type u_4
    inst✝⁸ : DivInvMonoid G
    inst✝⁷ : TopologicalSpace G
    inst✝⁶ : ChartedSpace H G
    inst✝⁵ : SmoothMul I G
    E' : Type u_5
    inst✝⁴ : NormedAddCommGroup E'
    inst✝³ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝² : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M : Type u_7
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H' M
    f : M → G
    s : Set M
    x : M
    n : ENat
    c : G
    hf : ContMDiffWithinAt I' I n f s x
    ⊢ ContMDiffWithinAt I' I n (fun x => HDiv.hDiv (f x) c) s x
  -/
  simpa only [div_eq_mul_inv] using hf.mul contMDiffWithinAt_const
  /-
    🎉 no goals
  -/


@[to_additive]
nonrec theorem ContMDiffAt.div_const (hf : ContMDiffAt I' I n f x) :
    ContMDiffAt I' I n (fun x ↦ f x / c) x :=
  hf.div_const c


@[to_additive]
theorem ContMDiffOn.div_const (hf : ContMDiffOn I' I n f s) :
    ContMDiffOn I' I n (fun x ↦ f x / c) s := fun x hx => (hf x hx).div_const c


@[to_additive]
theorem ContMDiff.div_const (hf : ContMDiff I' I n f) :
    ContMDiff I' I n (fun x ↦ f x / c) := fun x => (hf x).div_const c


@[deprecated (since := "2024-11-21")] alias SmoothWithinAt.div_const := ContMDiffWithinAt.div_const

@[deprecated (since := "2024-11-21")] alias SmoothAt.div_const := ContMDiffAt.div_const

@[deprecated (since := "2024-11-21")] alias SmoothOn.div_const := ContMDiffOn.div_const

@[deprecated (since := "2024-11-21")] alias Smooth.div_const := ContMDiff.div_const


