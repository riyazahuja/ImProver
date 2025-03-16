/-- An additive Lie group is a group and a smooth manifold at the same time in which
the addition and negation operations are smooth. -/
class LieAddGroup {𝕜 : Type*} [NontriviallyNormedField 𝕜] {H : Type*} [TopologicalSpace H]
    {E : Type*} [NormedAddCommGroup E] [NormedSpace 𝕜 E] (I : ModelWithCorners 𝕜 E H) (G : Type*)
    [AddGroup G] [TopologicalSpace G] [ChartedSpace H G] extends SmoothAdd I G : Prop where
  /-- Negation is smooth in an additive Lie group. -/
  smooth_neg : ContMDiff I I ⊤ fun a : G => -a

-- See note [Design choices about smooth algebraic structures]

/-- A (multiplicative) Lie group is a group and a smooth manifold at the same time in which
the multiplication and inverse operations are smooth. -/
@[to_additive]
class LieGroup {𝕜 : Type*} [NontriviallyNormedField 𝕜] {H : Type*} [TopologicalSpace H]
    {E : Type*} [NormedAddCommGroup E] [NormedSpace 𝕜 E] (I : ModelWithCorners 𝕜 E H) (G : Type*)
    [Group G] [TopologicalSpace G] [ChartedSpace H G] extends SmoothMul I G : Prop where
  /-- Inversion is smooth in a Lie group. -/
  smooth_inv : ContMDiff I I ⊤ fun a : G => a⁻¹


/-- In a Lie group, inversion is a smooth map. -/
@[to_additive "In an additive Lie group, inversion is a smooth map."]
theorem contMDiff_inv : ContMDiff I I ⊤ fun x : G => x⁻¹ :=
  LieGroup.smooth_inv


@[deprecated (since := "2024-11-21")] alias smooth_inv := contMDiff_inv

@[deprecated (since := "2024-11-21")] alias smooth_neg := contMDiff_neg


include I in
/-- A Lie group is a topological group. This is not an instance for technical reasons,
see note [Design choices about smooth algebraic structures]. -/
@[to_additive "An additive Lie group is an additive topological group. This is not an instance for
technical reasons, see note [Design choices about smooth algebraic structures]."]
theorem topologicalGroup_of_lieGroup : TopologicalGroup G :=
  { continuousMul_of_smooth I with continuous_inv := (contMDiff_inv I).continuous }


@[to_additive]
theorem ContMDiffWithinAt.inv {f : M → G} {s : Set M} {x₀ : M}
    (hf : ContMDiffWithinAt I' I n f s x₀) : ContMDiffWithinAt I' I n (fun x => (f x)⁻¹) s x₀ :=
  ((contMDiff_inv I).of_le le_top).contMDiffAt.contMDiffWithinAt.comp x₀ hf <| Set.mapsTo_univ _ _


@[to_additive]
theorem ContMDiffAt.inv {f : M → G} {x₀ : M} (hf : ContMDiffAt I' I n f x₀) :
    ContMDiffAt I' I n (fun x => (f x)⁻¹) x₀ :=
  ((contMDiff_inv I).of_le le_top).contMDiffAt.comp x₀ hf


@[to_additive]
theorem ContMDiffOn.inv {f : M → G} {s : Set M} (hf : ContMDiffOn I' I n f s) :
    ContMDiffOn I' I n (fun x => (f x)⁻¹) s := fun x hx => (hf x hx).inv


@[to_additive]
theorem ContMDiff.inv {f : M → G} (hf : ContMDiff I' I n f) : ContMDiff I' I n fun x => (f x)⁻¹ :=
  fun x => (hf x).inv


@[deprecated (since := "2024-11-21")] alias SmoothWithinAt.inv := ContMDiffWithinAt.inv

@[deprecated (since := "2024-11-21")] alias SmoothAt.inv := ContMDiffAt.inv

@[deprecated (since := "2024-11-21")] alias SmoothOn.inv := ContMDiffOn.inv

@[deprecated (since := "2024-11-21")] alias Smooth.inv := ContMDiff.inv


@[deprecated (since := "2024-11-21")] alias SmoothWithinAt.neg := ContMDiffWithinAt.neg

@[deprecated (since := "2024-11-21")] alias SmoothAt.neg := ContMDiffAt.neg

@[deprecated (since := "2024-11-21")] alias SmoothOn.neg := ContMDiffOn.neg

@[deprecated (since := "2024-11-21")] alias Smooth.neg := ContMDiff.neg


@[to_additive]
theorem ContMDiffWithinAt.div {f g : M → G} {s : Set M} {x₀ : M}
    (hf : ContMDiffWithinAt I' I n f s x₀) (hg : ContMDiffWithinAt I' I n g s x₀) :
    ContMDiffWithinAt I' I n (fun x => f x / g x) s x₀ := by
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
    inst✝⁸ : TopologicalSpace G
    inst✝⁷ : ChartedSpace H G
    inst✝⁶ : Group G
    inst✝⁵ : LieGroup I G
    E' : Type u_5
    inst✝⁴ : NormedAddCommGroup E'
    inst✝³ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝² : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M : Type u_7
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H' M
    n : ENat
    f g : M → G
    s : Set M
    x₀ : M
    hf : ContMDiffWithinAt I' I n f s x₀
    hg : ContMDiffWithinAt I' I n g s x₀
    ⊢ ContMDiffWithinAt I' I n (fun x => HDiv.hDiv (f x) (g x)) s x₀
  -/
  simp_rw [div_eq_mul_inv]; exact hf.mul hg.inv
                            /-
                              🎉 no goals
                            -/


@[to_additive]
theorem ContMDiffAt.div {f g : M → G} {x₀ : M} (hf : ContMDiffAt I' I n f x₀)
    (hg : ContMDiffAt I' I n g x₀) : ContMDiffAt I' I n (fun x => f x / g x) x₀ := by
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
    inst✝⁸ : TopologicalSpace G
    inst✝⁷ : ChartedSpace H G
    inst✝⁶ : Group G
    inst✝⁵ : LieGroup I G
    E' : Type u_5
    inst✝⁴ : NormedAddCommGroup E'
    inst✝³ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝² : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M : Type u_7
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H' M
    n : ENat
    f g : M → G
    x₀ : M
    hf : ContMDiffAt I' I n f x₀
    hg : ContMDiffAt I' I n g x₀
    ⊢ ContMDiffAt I' I n (fun x => HDiv.hDiv (f x) (g x)) x₀
  -/
  simp_rw [div_eq_mul_inv]; exact hf.mul hg.inv
                            /-
                              🎉 no goals
                            -/


@[to_additive]
theorem ContMDiffOn.div {f g : M → G} {s : Set M} (hf : ContMDiffOn I' I n f s)
    (hg : ContMDiffOn I' I n g s) : ContMDiffOn I' I n (fun x => f x / g x) s := by
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
    inst✝⁸ : TopologicalSpace G
    inst✝⁷ : ChartedSpace H G
    inst✝⁶ : Group G
    inst✝⁵ : LieGroup I G
    E' : Type u_5
    inst✝⁴ : NormedAddCommGroup E'
    inst✝³ : NormedSpace 𝕜 E'
    H' : Type u_6
    inst✝² : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E' H'
    M : Type u_7
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H' M
    n : ENat
    f g : M → G
    s : Set M
    hf : ContMDiffOn I' I n f s
    hg : ContMDiffOn I' I n g s
    ⊢ ContMDiffOn I' I n (fun x => HDiv.hDiv (f x) (g x)) s
  -/
  simp_rw [div_eq_mul_inv]; exact hf.mul hg.inv
                            /-
                              🎉 no goals
                            -/


@[to_additive]
theorem ContMDiff.div {f g : M → G} (hf : ContMDiff I' I n f) (hg : ContMDiff I' I n g) :
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
                                                inst✝⁸ : TopologicalSpace G
                                                inst✝⁷ : ChartedSpace H G
                                                inst✝⁶ : Group G
                                                inst✝⁵ : LieGroup I G
                                                E' : Type u_5
                                                inst✝⁴ : NormedAddCommGroup E'
                                                inst✝³ : NormedSpace 𝕜 E'
                                                H' : Type u_6
                                                inst✝² : TopologicalSpace H'
                                                I' : ModelWithCorners 𝕜 E' H'
                                                M : Type u_7
                                                inst✝¹ : TopologicalSpace M
                                                inst✝ : ChartedSpace H' M
                                                n : ENat
                                                f g : M → G
                                                hf : ContMDiff I' I n f
                                                hg : ContMDiff I' I n g
                                                ⊢ ContMDiff I' I n fun x => HDiv.hDiv (f x) (g x)
                                              -/
    ContMDiff I' I n fun x => f x / g x := by simp_rw [div_eq_mul_inv]; exact hf.mul hg.inv
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


@[deprecated (since := "2024-11-21")] alias SmoothWithinAt.div := ContMDiffWithinAt.div

@[deprecated (since := "2024-11-21")] alias SmoothAt.div := ContMDiffAt.div

@[deprecated (since := "2024-11-21")] alias SmoothOn.div := ContMDiffOn.div

@[deprecated (since := "2024-11-21")] alias Smooth.div := ContMDiff.div


@[deprecated (since := "2024-11-21")] alias SmoothWithinAt.sub := ContMDiffWithinAt.sub

@[deprecated (since := "2024-11-21")] alias SmoothAt.sub := ContMDiffAt.sub

@[deprecated (since := "2024-11-21")] alias SmoothOn.sub := ContMDiffOn.sub

@[deprecated (since := "2024-11-21")] alias Smooth.sub := ContMDiff.sub


@[to_additive]
instance {𝕜 : Type*} [NontriviallyNormedField 𝕜] {H : Type*} [TopologicalSpace H] {E : Type*}
    [NormedAddCommGroup E] [NormedSpace 𝕜 E] {I : ModelWithCorners 𝕜 E H} {G : Type*}
    [TopologicalSpace G] [ChartedSpace H G] [Group G] [LieGroup I G] {E' : Type*}
    [NormedAddCommGroup E'] [NormedSpace 𝕜 E'] {H' : Type*} [TopologicalSpace H']
    {I' : ModelWithCorners 𝕜 E' H'} {G' : Type*} [TopologicalSpace G'] [ChartedSpace H' G']
    [Group G'] [LieGroup I' G'] : LieGroup (I.prod I') (G × G') :=
  { SmoothMul.prod _ _ _ _ with smooth_inv := contMDiff_fst.inv.prod_mk contMDiff_snd.inv }


instance normedSpaceLieAddGroup {𝕜 : Type*} [NontriviallyNormedField 𝕜] {E : Type*}
    [NormedAddCommGroup E] [NormedSpace 𝕜 E] : LieAddGroup 𝓘(𝕜, E) E where
  smooth_neg := contDiff_neg.contMDiff


/-- A smooth manifold with `0` and `Inv` such that `fun x ↦ x⁻¹` is smooth at all nonzero points.
Any complete normed (semi)field has this property. -/
class SmoothInv₀ {𝕜 : Type*} [NontriviallyNormedField 𝕜] {H : Type*} [TopologicalSpace H]
    {E : Type*} [NormedAddCommGroup E] [NormedSpace 𝕜 E] (I : ModelWithCorners 𝕜 E H) (G : Type*)
    [Inv G] [Zero G] [TopologicalSpace G] [ChartedSpace H G] : Prop where
  /-- Inversion is smooth away from `0`. -/
  smoothAt_inv₀ : ∀ ⦃x : G⦄, x ≠ 0 → ContMDiffAt I I ⊤ (fun y ↦ y⁻¹) x


instance {𝕜 : Type*} [NontriviallyNormedField 𝕜] : SmoothInv₀ 𝓘(𝕜) 𝕜 where
  smoothAt_inv₀ x hx := by
    /-
      𝕜 : Type u_1
      inst✝ : NontriviallyNormedField 𝕜
      x : 𝕜
      hx : Ne x 0
      ⊢ ContMDiffAt (modelWithCornersSelf 𝕜 𝕜) (modelWithCornersSelf 𝕜 𝕜) Top.top (f …
    -/
    change ContMDiffAt 𝓘(𝕜) 𝓘(𝕜) ⊤ Inv.inv x
    /-
      𝕜 : Type u_1
      inst✝ : NontriviallyNormedField 𝕜
      x : 𝕜
      hx : Ne x 0
      ⊢ ContMDiffAt (modelWithCornersSelf 𝕜 𝕜) (modelWithCornersSelf 𝕜 𝕜) Top.top In …
    -/
    rw [contMDiffAt_iff_contDiffAt]
    /-
      𝕜 : Type u_1
      inst✝ : NontriviallyNormedField 𝕜
      x : 𝕜
      hx : Ne x 0
      ⊢ ContDiffAt 𝕜 (↑Top.top) Inv.inv x
    -/
    exact contDiffAt_inv 𝕜 hx
    /-
      🎉 no goals
    -/


theorem contMDiffAt_inv₀ {x : G} (hx : x ≠ 0) : ContMDiffAt I I ⊤ (fun y ↦ y⁻¹) x :=
  SmoothInv₀.smoothAt_inv₀ hx


@[deprecated (since := "2024-11-21")] alias smoothAt_inv₀ := contMDiffAt_inv₀


include I in
/-- In a manifold with smooth inverse away from `0`, the inverse is continuous away from `0`.
This is not an instance for technical reasons, see
note [Design choices about smooth algebraic structures]. -/
theorem hasContinuousInv₀_of_hasSmoothInv₀ : HasContinuousInv₀ G :=
  { continuousAt_inv₀ := fun _ hx ↦ (contMDiffAt_inv₀ (I := I) hx).continuousAt }


theorem contMDiffOn_inv₀ : ContMDiffOn I I ⊤ (Inv.inv : G → G) {0}ᶜ := fun _x hx =>
  (contMDiffAt_inv₀ hx).contMDiffWithinAt


@[deprecated (since := "2024-11-21")] alias smoothOn_inv₀ := contMDiffOn_inv₀

@[deprecated (since := "2024-11-21")] alias SmoothOn_inv₀ := contMDiffOn_inv₀


theorem ContMDiffWithinAt.inv₀ (hf : ContMDiffWithinAt I' I n f s a) (ha : f a ≠ 0) :
    ContMDiffWithinAt I' I n (fun x => (f x)⁻¹) s a :=
  ((contMDiffAt_inv₀ ha).of_le le_top).comp_contMDiffWithinAt a hf


theorem ContMDiffAt.inv₀ (hf : ContMDiffAt I' I n f a) (ha : f a ≠ 0) :
    ContMDiffAt I' I n (fun x ↦ (f x)⁻¹) a :=
  ((contMDiffAt_inv₀ ha).of_le le_top).comp a hf


theorem ContMDiff.inv₀ (hf : ContMDiff I' I n f) (h0 : ∀ x, f x ≠ 0) :
    ContMDiff I' I n (fun x ↦ (f x)⁻¹) :=
  fun x ↦ ContMDiffAt.inv₀ (hf x) (h0 x)


theorem ContMDiffOn.inv₀ (hf : ContMDiffOn I' I n f s) (h0 : ∀ x ∈ s, f x ≠ 0) :
    ContMDiffOn I' I n (fun x => (f x)⁻¹) s :=
  fun x hx ↦ ContMDiffWithinAt.inv₀ (hf x hx) (h0 x hx)


@[deprecated (since := "2024-11-21")] alias SmoothWithinAt.inv₀ := ContMDiffWithinAt.inv₀

@[deprecated (since := "2024-11-21")] alias SmoothAt.inv₀ := ContMDiffAt.inv₀

@[deprecated (since := "2024-11-21")] alias SmoothOn.inv₀ := ContMDiffOn.inv₀

@[deprecated (since := "2024-11-21")] alias Smooth.inv₀ := ContMDiff.inv₀


theorem ContMDiffWithinAt.div₀
    (hf : ContMDiffWithinAt I' I n f s a) (hg : ContMDiffWithinAt I' I n g s a) (h₀ : g a ≠ 0) :
    ContMDiffWithinAt I' I n (f / g) s a := by
  /-
    𝕜 : Type u_1
    inst✝¹³ : NontriviallyNormedField 𝕜
    H : Type u_2
    inst✝¹² : TopologicalSpace H
    E : Type u_3
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedSpace 𝕜 E
    I : ModelWithCorners 𝕜 E H
    G : Type u_4
    inst✝⁹ : TopologicalSpace G
    inst✝⁸ : ChartedSpace H G
    inst✝⁷ : GroupWithZero G
    inst✝⁶ : SmoothInv₀ I G
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
    f g : M → G
    s : Set M
    a : M
    n : ENat
    hf : ContMDiffWithinAt I' I n f s a
    hg : ContMDiffWithinAt I' I n g s a
    h₀ : Ne (g a) 0
    ⊢ ContMDiffWithinAt I' I n (HDiv.hDiv f g) s a
  -/
  simpa [div_eq_mul_inv] using hf.mul (hg.inv₀ h₀)
  /-
    🎉 no goals
  -/


theorem ContMDiffOn.div₀ (hf : ContMDiffOn I' I n f s) (hg : ContMDiffOn I' I n g s)
    (h₀ : ∀ x ∈ s, g x ≠ 0) : ContMDiffOn I' I n (f / g) s := by
  /-
    𝕜 : Type u_1
    inst✝¹³ : NontriviallyNormedField 𝕜
    H : Type u_2
    inst✝¹² : TopologicalSpace H
    E : Type u_3
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedSpace 𝕜 E
    I : ModelWithCorners 𝕜 E H
    G : Type u_4
    inst✝⁹ : TopologicalSpace G
    inst✝⁸ : ChartedSpace H G
    inst✝⁷ : GroupWithZero G
    inst✝⁶ : SmoothInv₀ I G
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
    f g : M → G
    s : Set M
    n : ENat
    hf : ContMDiffOn I' I n f s
    hg : ContMDiffOn I' I n g s
    h₀ : ∀ (x : M), Membership.mem s x → Ne (g x) 0
    ⊢ ContMDiffOn I' I n (HDiv.hDiv f g) s
  -/
  simpa [div_eq_mul_inv] using hf.mul (hg.inv₀ h₀)
  /-
    🎉 no goals
  -/


theorem ContMDiffAt.div₀ (hf : ContMDiffAt I' I n f a) (hg : ContMDiffAt I' I n g a)
    (h₀ : g a ≠ 0) : ContMDiffAt I' I n (f / g) a := by
  /-
    𝕜 : Type u_1
    inst✝¹³ : NontriviallyNormedField 𝕜
    H : Type u_2
    inst✝¹² : TopologicalSpace H
    E : Type u_3
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedSpace 𝕜 E
    I : ModelWithCorners 𝕜 E H
    G : Type u_4
    inst✝⁹ : TopologicalSpace G
    inst✝⁸ : ChartedSpace H G
    inst✝⁷ : GroupWithZero G
    inst✝⁶ : SmoothInv₀ I G
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
    f g : M → G
    a : M
    n : ENat
    hf : ContMDiffAt I' I n f a
    hg : ContMDiffAt I' I n g a
    h₀ : Ne (g a) 0
    ⊢ ContMDiffAt I' I n (HDiv.hDiv f g) a
  -/
  simpa [div_eq_mul_inv] using hf.mul (hg.inv₀ h₀)
  /-
    🎉 no goals
  -/


theorem ContMDiff.div₀ (hf : ContMDiff I' I n f) (hg : ContMDiff I' I n g) (h₀ : ∀ x, g x ≠ 0) :
                                   /-
                                     𝕜 : Type u_1
                                     inst✝¹³ : NontriviallyNormedField 𝕜
                                     H : Type u_2
                                     inst✝¹² : TopologicalSpace H
                                     E : Type u_3
                                     inst✝¹¹ : NormedAddCommGroup E
                                     inst✝¹⁰ : NormedSpace 𝕜 E
                                     I : ModelWithCorners 𝕜 E H
                                     G : Type u_4
                                     inst✝⁹ : TopologicalSpace G
                                     inst✝⁸ : ChartedSpace H G
                                     inst✝⁷ : GroupWithZero G
                                     inst✝⁶ : SmoothInv₀ I G
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
                                     f g : M → G
                                     n : ENat
                                     hf : ContMDiff I' I n f
                                     hg : ContMDiff I' I n g
                                     h₀ : ∀ (x : M), Ne (g x) 0
                                     ⊢ ContMDiff I' I n (HDiv.hDiv f g)
                                   -/
    ContMDiff I' I n (f / g) := by simpa only [div_eq_mul_inv] using hf.mul (hg.inv₀ h₀)
                                   /-
                                     🎉 no goals
                                   -/


@[deprecated (since := "2024-11-21")] alias SmoothWithinAt.div₀ := ContMDiffWithinAt.div₀

@[deprecated (since := "2024-11-21")] alias SmoothAt.div₀ := ContMDiffAt.div₀

@[deprecated (since := "2024-11-21")] alias SmoothOn.div₀ := ContMDiffOn.div₀

@[deprecated (since := "2024-11-21")] alias Smooth.div₀ := ContMDiff.div₀


