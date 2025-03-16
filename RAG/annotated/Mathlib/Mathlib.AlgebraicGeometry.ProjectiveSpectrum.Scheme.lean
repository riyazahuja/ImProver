/-- `Proj` as a locally ringed space -/
local notation3 "Proj" => Proj.toLocallyRingedSpace 𝒜


/-- The underlying topological space of `Proj` -/
local notation3 "Proj.T" => PresheafedSpace.carrier <| SheafedSpace.toPresheafedSpace
  <| LocallyRingedSpace.toSheafedSpace <| Proj.toLocallyRingedSpace 𝒜


/-- `Proj` restrict to some open set -/
macro "Proj| " U:term : term =>
  `((Proj.toLocallyRingedSpace 𝒜).restrict
    (Opens.isOpenEmbedding (X := Proj.T) ($U : Opens Proj.T)))


/-- the underlying topological space of `Proj` restricted to some open set -/
local notation "Proj.T| " U => PresheafedSpace.carrier <| SheafedSpace.toPresheafedSpace
  <| LocallyRingedSpace.toSheafedSpace
    <| (LocallyRingedSpace.restrict Proj (Opens.isOpenEmbedding (X := Proj.T) (U : Opens Proj.T)))


/-- basic open sets in `Proj` -/
local notation "pbo " x => ProjectiveSpectrum.basicOpen 𝒜 x


/-- basic open sets in `Spec` -/
local notation "sbo " f => PrimeSpectrum.basicOpen f


/-- `Spec` as a locally ringed space -/
local notation3 "Spec " ring => Spec.locallyRingedSpaceObj (CommRingCat.of ring)


/-- the underlying topological space of `Spec` -/
local notation "Spec.T " ring =>
  (Spec.locallyRingedSpaceObj (CommRingCat.of ring)).toSheafedSpace.toPresheafedSpace.1


local notation3 "A⁰_ " f => HomogeneousLocalization.Away 𝒜 f


/--
For any `x` in `Proj| (pbo f)`, the corresponding ideal in `Spec A⁰_f`. This fact that this ideal
is prime is proven in `TopComponent.Forward.toFun`-/
def carrier : Ideal (A⁰_ f) :=
  Ideal.comap (algebraMap (A⁰_ f) (Away f))
    (x.val.asHomogeneousIdeal.toIdeal.map (algebraMap A (Away f)))


@[simp]
theorem mk_mem_carrier (z : HomogeneousLocalization.NumDenSameDeg 𝒜 (.powers f)) :
    HomogeneousLocalization.mk z ∈ carrier x ↔ z.num.1 ∈ x.1.asHomogeneousIdeal := by
  rw [carrier, Ideal.mem_comap, HomogeneousLocalization.algebraMap_apply,
    HomogeneousLocalization.val_mk, Localization.mk_eq_mk', IsLocalization.mk'_eq_mul_mk'_one,
    mul_comm, Ideal.unit_mul_mem_iff_mem, ← Ideal.mem_comap,
    IsLocalization.comap_map_of_isPrime_disjoint (.powers f)]
    /-
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      f : A
      x : ↑((AlgebraicGeometry.Proj.toLocallyRingedSpace 𝒜).restrict ⋯).toTopCat
      z : HomogeneousLocalization.NumDenSameDeg 𝒜 (Submonoid.powers f)
      ⊢ Iff (Membership.mem (↑x).asHomogeneousIdeal.toIdeal ↑z.num) (Membership.mem  …
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case hI
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      f : A
      x : ↑((AlgebraicGeometry.Proj.toLocallyRingedSpace 𝒜).restrict ⋯).toTopCat
      z : HomogeneousLocalization.NumDenSameDeg 𝒜 (Submonoid.powers f)
      ⊢ (↑x).asHomogeneousIdeal.toIdeal.IsPrime
    -/
  · infer_instance
    /-
      🎉 no goals
    -/
    /-
      case hM
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      f : A
      x : ↑((AlgebraicGeometry.Proj.toLocallyRingedSpace 𝒜).restrict ⋯).toTopCat
      z : HomogeneousLocalization.NumDenSameDeg 𝒜 (Submonoid.powers f)
      ⊢ Disjoint ↑(Submonoid.powers f) ↑(↑x).asHomogeneousIdeal.toIdeal
    -/
  · exact (disjoint_powers_iff_not_mem _ (Ideal.IsPrime.isRadical inferInstance)).mpr x.2
    /-
      🎉 no goals
    -/
    /-
      case hy
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      f : A
      x : ↑((AlgebraicGeometry.Proj.toLocallyRingedSpace 𝒜).restrict ⋯).toTopCat
      z : HomogeneousLocalization.NumDenSameDeg 𝒜 (Submonoid.powers f)
      ⊢ IsUnit (IsLocalization.mk' (Localization (Submonoid.powers f)) 1 ⟨↑z.den, ⋯⟩)
    -/
  · exact isUnit_of_invertible _
    /-
      🎉 no goals
    -/


theorem isPrime_carrier : Ideal.IsPrime (carrier x) := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    x : ↑((AlgebraicGeometry.Proj.toLocallyRingedSpace 𝒜).restrict ⋯).toTopCat
    ⊢ (AlgebraicGeometry.ProjIsoSpecTopComponent.ToSpec.carrier x).IsPrime
  -/
  refine Ideal.IsPrime.comap _ (hK := ?_)
  exact IsLocalization.isPrime_of_isPrime_disjoint
    (Submonoid.powers f) _ _ inferInstance
    ((disjoint_powers_iff_not_mem _ (Ideal.IsPrime.isRadical inferInstance)).mpr x.2)


/-- The function between the basic open set `D(f)` in `Proj` to the corresponding basic open set in
`Spec A⁰_f`. This is bundled into a continuous map in `TopComponent.forward`.
-/
@[simps (config := .lemmasOnly)]
def toFun (x : Proj.T| pbo f) : Spec.T A⁰_ f :=
  ⟨carrier x, isPrime_carrier x⟩

/-
The preimage of basic open set `D(a/f^n)` in `Spec A⁰_f` under the forward map from `Proj A` to
`Spec A⁰_f` is the basic open set `D(a) ∩ D(f)` in `Proj A`. This lemma is used to prove that the
forward map is continuous.
-/

theorem preimage_basicOpen (z : HomogeneousLocalization.NumDenSameDeg 𝒜 (.powers f)) :
    toFun f ⁻¹' (sbo (HomogeneousLocalization.mk z) : Set (PrimeSpectrum (A⁰_ f))) =
      Subtype.val ⁻¹' (pbo z.num.1 : Set (ProjectiveSpectrum 𝒜)) :=
  Set.ext fun y ↦ (mk_mem_carrier y z).not


/-- The continuous function from the basic open set `D(f)` in `Proj`
to the corresponding basic open set in `Spec A⁰_f`. -/
@[simps! (config := .lemmasOnly) apply_asIdeal]
def toSpec (f : A) : (Proj.T| pbo f) ⟶ Spec.T A⁰_ f where
  toFun := ToSpec.toFun f
  continuous_toFun := by
    /-
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      f : A
      ⊢ Continuous (AlgebraicGeometry.ProjIsoSpecTopComponent.ToSpec.toFun f)
    -/
    rw [PrimeSpectrum.isTopologicalBasis_basic_opens.continuous_iff]
    /-
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      f : A
      ⊢ ∀ (s : Set (PrimeSpectrum ↑(CommRingCat.of (HomogeneousLocalization.Away 𝒜 f …
    -/
    rintro _ ⟨x, rfl⟩
    /-
      case intro
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      f : A
      x : ↑(CommRingCat.of (HomogeneousLocalization.Away 𝒜 f))
      ⊢ IsOpen (Set.preimage (AlgebraicGeometry.ProjIsoSpecTopComponent.ToSpec.toFun …
    -/
    obtain ⟨x, rfl⟩ := Quotient.mk''_surjective x
    /-
      case intro.intro
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      f : A
      x : HomogeneousLocalization.NumDenSameDeg 𝒜 (Submonoid.powers f)
      ⊢ IsOpen (Set.preimage (AlgebraicGeometry.ProjIsoSpecTopComponent.ToSpec.toFun …
    -/
    rw [ToSpec.preimage_basicOpen]
    /-
      case intro.intro
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      f : A
      x : HomogeneousLocalization.NumDenSameDeg 𝒜 (Submonoid.powers f)
      ⊢ IsOpen (Set.preimage Subtype.val ↑(ProjectiveSpectrum.basicOpen 𝒜 ↑x.num))
    -/
    exact (pbo x.num).2.preimage continuous_subtype_val
    /-
      🎉 no goals
    -/


variable {𝒜} in
lemma toSpec_preimage_basicOpen {f} (z : HomogeneousLocalization.NumDenSameDeg 𝒜 (.powers f)) :
    toSpec 𝒜 f ⁻¹' (sbo (HomogeneousLocalization.mk z) : Set (PrimeSpectrum (A⁰_ f))) =
      Subtype.val ⁻¹' (pbo z.num.1 : Set (ProjectiveSpectrum 𝒜)) :=
  ToSpec.preimage_basicOpen f z


macro "mem_tac_aux" : tactic =>
  `(tactic| first | exact pow_mem_graded _ (Submodule.coe_mem _) | exact natCast_mem_graded _ _ |
    exact pow_mem_graded _ f_deg)


macro "mem_tac" : tactic =>
  `(tactic| first | mem_tac_aux |
    repeat (all_goals (apply SetLike.GradedMonoid.toGradedMul.mul_mem)); mem_tac_aux)


/-- The function from `Spec A⁰_f` to `Proj|D(f)` is defined by `q ↦ {a | aᵢᵐ/fⁱ ∈ q}`, i.e. sending
`q` a prime ideal in `A⁰_f` to the homogeneous prime relevant ideal containing only and all the
elements `a : A` such that for every `i`, the degree 0 element formed by dividing the `m`-th power
of the `i`-th projection of `a` by the `i`-th power of the degree-`m` homogeneous element `f`,
lies in `q`.

The set `{a | aᵢᵐ/fⁱ ∈ q}`
* is an ideal, as proved in `carrier.asIdeal`;
* is homogeneous, as proved in `carrier.asHomogeneousIdeal`;
* is prime, as proved in `carrier.asIdeal.prime`;
* is relevant, as proved in `carrier.relevant`.
-/
def carrier (f_deg : f ∈ 𝒜 m) (q : Spec.T A⁰_ f) : Set A :=
                                                                    /-
                                                                      R : Type u_1
                                                                      A : Type u_2
                                                                      inst✝³ : CommRing R
                                                                      inst✝² : CommRing A
                                                                      inst✝¹ : Algebra R A
                                                                      𝒜 : Nat → Submodule R A
                                                                      inst✝ : GradedAlgebra 𝒜
                                                                      f : A
                                                                      m : Nat
                                                                      f_deg✝ f_deg : Membership.mem (𝒜 m) f
                                                                      q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
                                                                      a : A
                                                                      i : Nat
                                                                      ⊢ Membership.mem (𝒜 (HMul.hMul m i)) (HPow.hPow ((GradedAlgebra.proj 𝒜 i) a) m)
                                                                    -/
  {a | ∀ i, (HomogeneousLocalization.mk ⟨m * i, ⟨proj 𝒜 i a ^ m, by rw [← smul_eq_mul]; mem_tac⟩,
                                                                                        /-
                                                                                          🎉 no goals
                                                                                        -/
                         /-
                           R : Type u_1
                           A : Type u_2
                           inst✝³ : CommRing R
                           inst✝² : CommRing A
                           inst✝¹ : Algebra R A
                           𝒜 : Nat → Submodule R A
                           inst✝ : GradedAlgebra 𝒜
                           f : A
                           m : Nat
                           f_deg✝ f_deg : Membership.mem (𝒜 m) f
                           q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
                           a : A
                           i : Nat
                           ⊢ Membership.mem (𝒜 (HMul.hMul m i)) (HPow.hPow f i)
                         -/
              ⟨f ^ i, by rw [mul_comm]; mem_tac⟩, ⟨_, rfl⟩⟩ : A⁰_ f) ∈ q.1}
                                        /-
                                          🎉 no goals
                                        -/


theorem mem_carrier_iff (q : Spec.T A⁰_ f) (a : A) :
    a ∈ carrier f_deg q ↔ ∀ i, (HomogeneousLocalization.mk ⟨m * i, ⟨proj 𝒜 i a ^ m, by
      /-
        R : Type u_1
        A : Type u_2
        inst✝³ : CommRing R
        inst✝² : CommRing A
        inst✝¹ : Algebra R A
        𝒜 : Nat → Submodule R A
        inst✝ : GradedAlgebra 𝒜
        f : A
        m : Nat
        f_deg : Membership.mem (𝒜 m) f
        q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
        a : A
        i : Nat
        ⊢ Membership.mem (𝒜 (HMul.hMul m i)) (HPow.hPow ((GradedAlgebra.proj 𝒜 i) a) m)
      -/
      rw [← smul_eq_mul]; mem_tac⟩,
                          /-
                            🎉 no goals
                          -/
                 /-
                   R : Type u_1
                   A : Type u_2
                   inst✝³ : CommRing R
                   inst✝² : CommRing A
                   inst✝¹ : Algebra R A
                   𝒜 : Nat → Submodule R A
                   inst✝ : GradedAlgebra 𝒜
                   f : A
                   m : Nat
                   f_deg : Membership.mem (𝒜 m) f
                   q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
                   a : A
                   i : Nat
                   ⊢ Membership.mem (𝒜 (HMul.hMul m i)) (HPow.hPow f i)
                 -/
      ⟨f ^ i, by rw [mul_comm]; mem_tac⟩, ⟨_, rfl⟩⟩ : A⁰_ f) ∈ q.1 :=
                                /-
                                  🎉 no goals
                                -/
  Iff.rfl


theorem mem_carrier_iff' (q : Spec.T A⁰_ f) (a : A) :
    a ∈ carrier f_deg q ↔
      ∀ i, (Localization.mk (proj 𝒜 i a ^ m) ⟨f ^ i, ⟨i, rfl⟩⟩ : Localization.Away f) ∈
          algebraMap (HomogeneousLocalization.Away 𝒜 f) (Localization.Away f) '' { s | s ∈ q.1 } :=
  (mem_carrier_iff f_deg q a).trans
    (by
      /-
        R : Type u_1
        A : Type u_2
        inst✝³ : CommRing R
        inst✝² : CommRing A
        inst✝¹ : Algebra R A
        𝒜 : Nat → Submodule R A
        inst✝ : GradedAlgebra 𝒜
        f : A
        m : Nat
        f_deg : Membership.mem (𝒜 m) f
        q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
        a : A
        ⊢ Iff (∀ (i : Nat), Membership.mem q.asIdeal (HomogeneousLocalization.mk { deg …
      -/
      constructor <;> intro h i <;> specialize h i
        /-
          case mp
          R : Type u_1
          A : Type u_2
          inst✝³ : CommRing R
          inst✝² : CommRing A
          inst✝¹ : Algebra R A
          𝒜 : Nat → Submodule R A
          inst✝ : GradedAlgebra 𝒜
          f : A
          m : Nat
          f_deg : Membership.mem (𝒜 m) f
          q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
          a : A
          i : Nat
          h : Membership.mem q.asIdeal (HomogeneousLocalization.mk { deg := HMul.hMul m  …
          ⊢ Membership.mem (Set.image (⇑(algebraMap (HomogeneousLocalization.Away 𝒜 f) ( …
        -/
      · rw [Set.mem_image]; refine ⟨_, h, rfl⟩
                            /-
                              🎉 no goals
                            -/
        /-
          case mpr
          R : Type u_1
          A : Type u_2
          inst✝³ : CommRing R
          inst✝² : CommRing A
          inst✝¹ : Algebra R A
          𝒜 : Nat → Submodule R A
          inst✝ : GradedAlgebra 𝒜
          f : A
          m : Nat
          f_deg : Membership.mem (𝒜 m) f
          q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
          a : A
          i : Nat
          h : Membership.mem (Set.image (⇑(algebraMap (HomogeneousLocalization.Away 𝒜 f) …
          ⊢ Membership.mem q.asIdeal (HomogeneousLocalization.mk { deg := HMul.hMul m i, …
        -/
      · rw [Set.mem_image] at h; rcases h with ⟨x, h, hx⟩
        /-
          case mpr.intro.intro
          R : Type u_1
          A : Type u_2
          inst✝³ : CommRing R
          inst✝² : CommRing A
          inst✝¹ : Algebra R A
          𝒜 : Nat → Submodule R A
          inst✝ : GradedAlgebra 𝒜
          f : A
          m : Nat
          f_deg : Membership.mem (𝒜 m) f
          q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
          a : A
          i : Nat
          x : HomogeneousLocalization.Away 𝒜 f
          h : Membership.mem (setOf fun s => Membership.mem q.asIdeal s) x
          hx : Eq ((algebraMap (HomogeneousLocalization.Away 𝒜 f) (Localization.Away f)) …
          ⊢ Membership.mem q.asIdeal (HomogeneousLocalization.mk { deg := HMul.hMul m i, …
        -/
        change x ∈ q.asIdeal at h
        /-
          case mpr.intro.intro
          R : Type u_1
          A : Type u_2
          inst✝³ : CommRing R
          inst✝² : CommRing A
          inst✝¹ : Algebra R A
          𝒜 : Nat → Submodule R A
          inst✝ : GradedAlgebra 𝒜
          f : A
          m : Nat
          f_deg : Membership.mem (𝒜 m) f
          q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
          a : A
          i : Nat
          x : HomogeneousLocalization.Away 𝒜 f
          hx : Eq ((algebraMap (HomogeneousLocalization.Away 𝒜 f) (Localization.Away f)) …
          h : Membership.mem q.asIdeal x
          ⊢ Membership.mem q.asIdeal (HomogeneousLocalization.mk { deg := HMul.hMul m i, …
        -/
        convert h
        /-
          case h.e'_5
          R : Type u_1
          A : Type u_2
          inst✝³ : CommRing R
          inst✝² : CommRing A
          inst✝¹ : Algebra R A
          𝒜 : Nat → Submodule R A
          inst✝ : GradedAlgebra 𝒜
          f : A
          m : Nat
          f_deg : Membership.mem (𝒜 m) f
          q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
          a : A
          i : Nat
          x : HomogeneousLocalization.Away 𝒜 f
          hx : Eq ((algebraMap (HomogeneousLocalization.Away 𝒜 f) (Localization.Away f)) …
          h : Membership.mem q.asIdeal x
          ⊢ Eq (HomogeneousLocalization.mk { deg := HMul.hMul m i, num := ⟨HPow.hPow ((G …
        -/
        rw [HomogeneousLocalization.ext_iff_val, HomogeneousLocalization.val_mk]
        /-
          case h.e'_5
          R : Type u_1
          A : Type u_2
          inst✝³ : CommRing R
          inst✝² : CommRing A
          inst✝¹ : Algebra R A
          𝒜 : Nat → Submodule R A
          inst✝ : GradedAlgebra 𝒜
          f : A
          m : Nat
          f_deg : Membership.mem (𝒜 m) f
          q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
          a : A
          i : Nat
          x : HomogeneousLocalization.Away 𝒜 f
          hx : Eq ((algebraMap (HomogeneousLocalization.Away 𝒜 f) (Localization.Away f)) …
          h : Membership.mem q.asIdeal x
          ⊢ Eq (Localization.mk ↑{ deg := HMul.hMul m i, num := ⟨HPow.hPow ((GradedAlgeb …
        -/
        dsimp only [Subtype.coe_mk]; rw [← hx]; rfl)
                                                /-
                                                  🎉 no goals
                                                -/


theorem mem_carrier_iff_of_mem (hm : 0 < m) (q : Spec.T A⁰_ f) (a : A) {n} (hn : a ∈ 𝒜 n) :
    a ∈ carrier f_deg q ↔
      (HomogeneousLocalization.mk ⟨m * n, ⟨a ^ m, pow_mem_graded m hn⟩,
                   /-
                     R : Type u_1
                     A : Type u_2
                     inst✝³ : CommRing R
                     inst✝² : CommRing A
                     inst✝¹ : Algebra R A
                     𝒜 : Nat → Submodule R A
                     inst✝ : GradedAlgebra 𝒜
                     f : A
                     m : Nat
                     f_deg : Membership.mem (𝒜 m) f
                     hm : LT.lt 0 m
                     q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
                     a : A
                     n : Nat
                     hn : Membership.mem (𝒜 n) a
                     ⊢ Membership.mem (𝒜 (HMul.hMul m n)) (HPow.hPow f n)
                   -/
        ⟨f ^ n, by rw [mul_comm]; mem_tac⟩, ⟨_, rfl⟩⟩ : A⁰_ f) ∈ q.asIdeal := by
                                  /-
                                    🎉 no goals
                                  -/
  trans (HomogeneousLocalization.mk ⟨m * n, ⟨proj 𝒜 n a ^ m, by rw [← smul_eq_mul]; mem_tac⟩,
    ⟨f ^ n, by rw [mul_comm]; mem_tac⟩, ⟨_, rfl⟩⟩ : A⁰_ f) ∈ q.asIdeal
    /-
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      f : A
      m : Nat
      f_deg : Membership.mem (𝒜 m) f
      hm : LT.lt 0 m
      q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
      a : A
      n : Nat
      hn : Membership.mem (𝒜 n) a
      ⊢ Iff (Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carr …
    -/
  · refine ⟨fun h ↦ h n, fun h i ↦ if hi : i = n then hi ▸ h else ?_⟩
    /-
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      f : A
      m : Nat
      f_deg : Membership.mem (𝒜 m) f
      hm : LT.lt 0 m
      q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
      a : A
      n : Nat
      hn : Membership.mem (𝒜 n) a
      h : Membership.mem q.asIdeal (HomogeneousLocalization.mk { deg := HMul.hMul m  …
      i : Nat
      hi : Not (Eq i n)
      ⊢ Membership.mem q.asIdeal (HomogeneousLocalization.mk { deg := HMul.hMul m i, …
    -/
    convert zero_mem q.asIdeal
    /-
      case h.e'_5
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      f : A
      m : Nat
      f_deg : Membership.mem (𝒜 m) f
      hm : LT.lt 0 m
      q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
      a : A
      n : Nat
      hn : Membership.mem (𝒜 n) a
      h : Membership.mem q.asIdeal (HomogeneousLocalization.mk { deg := HMul.hMul m  …
      i : Nat
      hi : Not (Eq i n)
      ⊢ Eq (HomogeneousLocalization.mk { deg := HMul.hMul m i, num := ⟨HPow.hPow ((G …
    -/
    apply HomogeneousLocalization.val_injective
    simp only [proj_apply, decompose_of_mem_ne _ hn (Ne.symm hi), zero_pow hm.ne',
      HomogeneousLocalization.val_mk, Localization.mk_zero, HomogeneousLocalization.val_zero]
    /-
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      f : A
      m : Nat
      f_deg : Membership.mem (𝒜 m) f
      hm : LT.lt 0 m
      q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
      a : A
      n : Nat
      hn : Membership.mem (𝒜 n) a
      ⊢ Iff (Membership.mem q.asIdeal (HomogeneousLocalization.mk { deg := HMul.hMul …
    -/
  · simp only [proj_apply, decompose_of_mem_same _ hn]
    /-
      🎉 no goals
    -/


theorem mem_carrier_iff_of_mem_mul (hm : 0 < m)
    (q : Spec.T A⁰_ f) (a : A) {n} (hn : a ∈ 𝒜 (n * m)) :
    a ∈ carrier f_deg q ↔ (HomogeneousLocalization.mk ⟨m * n, ⟨a, mul_comm n m ▸ hn⟩,
                   /-
                     R : Type u_1
                     A : Type u_2
                     inst✝³ : CommRing R
                     inst✝² : CommRing A
                     inst✝¹ : Algebra R A
                     𝒜 : Nat → Submodule R A
                     inst✝ : GradedAlgebra 𝒜
                     f : A
                     m : Nat
                     f_deg : Membership.mem (𝒜 m) f
                     hm : LT.lt 0 m
                     q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
                     a : A
                     n : Nat
                     hn : Membership.mem (𝒜 (HMul.hMul n m)) a
                     ⊢ Membership.mem (𝒜 (HMul.hMul m n)) (HPow.hPow f n)
                   -/
        ⟨f ^ n, by rw [mul_comm]; mem_tac⟩, ⟨_, rfl⟩⟩ : A⁰_ f) ∈ q.asIdeal := by
                                  /-
                                    🎉 no goals
                                  -/
  rw [mem_carrier_iff_of_mem f_deg hm q a hn, iff_iff_eq, eq_comm,
    ← Ideal.IsPrime.pow_mem_iff_mem (α := A⁰_ f) inferInstance m hm]
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    m : Nat
    f_deg : Membership.mem (𝒜 m) f
    hm : LT.lt 0 m
    q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
    a : A
    n : Nat
    hn : Membership.mem (𝒜 (HMul.hMul n m)) a
    ⊢ Eq (Membership.mem q.asIdeal (HPow.hPow (HomogeneousLocalization.mk { deg := …
  -/
  congr 1
  /-
    case e_a
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    m : Nat
    f_deg : Membership.mem (𝒜 m) f
    hm : LT.lt 0 m
    q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
    a : A
    n : Nat
    hn : Membership.mem (𝒜 (HMul.hMul n m)) a
    ⊢ Eq (HPow.hPow (HomogeneousLocalization.mk { deg := HMul.hMul m n, num := ⟨a, …
  -/
  apply HomogeneousLocalization.val_injective
  simp only [HomogeneousLocalization.val_mk, HomogeneousLocalization.val_pow,
    Localization.mk_pow, pow_mul]
  /-
    case e_a.a
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    m : Nat
    f_deg : Membership.mem (𝒜 m) f
    hm : LT.lt 0 m
    q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
    a : A
    n : Nat
    hn : Membership.mem (𝒜 (HMul.hMul n m)) a
    ⊢ Eq (Localization.mk (HPow.hPow a m) (HPow.hPow ⟨HPow.hPow f n, ⋯⟩ m)) (Local …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem num_mem_carrier_iff (hm : 0 < m) (q : Spec.T A⁰_ f)
    (z : HomogeneousLocalization.NumDenSameDeg 𝒜 (.powers f)) :
    z.num.1 ∈ carrier f_deg q ↔ HomogeneousLocalization.mk z ∈ q.asIdeal := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    m : Nat
    f_deg : Membership.mem (𝒜 m) f
    hm : LT.lt 0 m
    q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
    z : HomogeneousLocalization.NumDenSameDeg 𝒜 (Submonoid.powers f)
    ⊢ Iff (Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carr …
  -/
  obtain ⟨n, hn : f ^ n = _⟩ := z.den_mem
  have : f ^ n ≠ 0 := fun e ↦ by
    have := HomogeneousLocalization.subsingleton 𝒜 (x := .powers f) ⟨n, e⟩
    exact IsEmpty.elim (inferInstanceAs (IsEmpty (PrimeSpectrum (A⁰_ f)))) q
  /-
    case intro
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    m : Nat
    f_deg : Membership.mem (𝒜 m) f
    hm : LT.lt 0 m
    q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
    z : HomogeneousLocalization.NumDenSameDeg 𝒜 (Submonoid.powers f)
    n : Nat
    hn : Eq (HPow.hPow f n) ↑z.den
    this : Ne (HPow.hPow f n) 0
    ⊢ Iff (Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carr …
  -/
  convert mem_carrier_iff_of_mem_mul f_deg hm q z.num.1 (n := n) ?_ using 2
    /-
      case h.e'_2.h.e'_5
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      f : A
      m : Nat
      f_deg : Membership.mem (𝒜 m) f
      hm : LT.lt 0 m
      q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
      z : HomogeneousLocalization.NumDenSameDeg 𝒜 (Submonoid.powers f)
      n : Nat
      hn : Eq (HPow.hPow f n) ↑z.den
      this : Ne (HPow.hPow f n) 0
      ⊢ Eq (HomogeneousLocalization.mk z) (HomogeneousLocalization.mk { deg := HMul. …
    -/
  · apply HomogeneousLocalization.val_injective; simp only [hn, HomogeneousLocalization.val_mk]
                                                 /-
                                                   🎉 no goals
                                                 -/
    /-
      case intro
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      f : A
      m : Nat
      f_deg : Membership.mem (𝒜 m) f
      hm : LT.lt 0 m
      q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
      z : HomogeneousLocalization.NumDenSameDeg 𝒜 (Submonoid.powers f)
      n : Nat
      hn : Eq (HPow.hPow f n) ↑z.den
      this : Ne (HPow.hPow f n) 0
      ⊢ Membership.mem (𝒜 (HMul.hMul n m)) ↑z.num
    -/
  · have := degree_eq_of_mem_mem 𝒜 (SetLike.pow_mem_graded n f_deg) (hn.symm ▸ z.den.2) this
    /-
      case intro
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      f : A
      m : Nat
      f_deg : Membership.mem (𝒜 m) f
      hm : LT.lt 0 m
      q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
      z : HomogeneousLocalization.NumDenSameDeg 𝒜 (Submonoid.powers f)
      n : Nat
      hn : Eq (HPow.hPow f n) ↑z.den
      this✝ : Ne (HPow.hPow f n) 0
      this : Eq (HSMul.hSMul n m) z.deg
      ⊢ Membership.mem (𝒜 (HMul.hMul n m)) ↑z.num
    -/
    rw [← smul_eq_mul, this]; exact z.num.2
                              /-
                                🎉 no goals
                              -/


theorem carrier.add_mem (q : Spec.T A⁰_ f) {a b : A} (ha : a ∈ carrier f_deg q)
    (hb : b ∈ carrier f_deg q) : a + b ∈ carrier f_deg q := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    m : Nat
    f_deg : Membership.mem (𝒜 m) f
    q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
    a b : A
    ha : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrie …
    hb : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrie …
    ⊢ Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrier f …
  -/
  refine fun i => (q.2.mem_or_mem ?_).elim id id
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    m : Nat
    f_deg : Membership.mem (𝒜 m) f
    q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
    a b : A
    ha : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrie …
    hb : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrie …
    i : Nat
    ⊢ Membership.mem q.asIdeal (HMul.hMul (HomogeneousLocalization.mk { deg := HMu …
  -/
  change (HomogeneousLocalization.mk ⟨_, _, _, _⟩ : A⁰_ f) ∈ q.1; dsimp only [Subtype.coe_mk]
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    m : Nat
    f_deg : Membership.mem (𝒜 m) f
    q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
    a b : A
    ha : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrie …
    hb : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrie …
    i : Nat
    ⊢ Membership.mem q.asIdeal (HomogeneousLocalization.mk { deg := HAdd.hAdd (HMu …
  -/
  simp_rw [← pow_add, map_add, add_pow, mul_comm, ← nsmul_eq_mul]
  let g : ℕ → A⁰_ f := fun j => (m + m).choose j •
      if h2 : m + m < j then (0 : A⁰_ f)
      else
        -- Porting note: inlining `l`, `r` causes a "can't synth HMul A⁰_ f A⁰_ f ?" error
        if h1 : j ≤ m then
          letI l : A⁰_ f := HomogeneousLocalization.mk
            ⟨m * i, ⟨proj 𝒜 i a ^ j * proj 𝒜 i b ^ (m - j), ?_⟩,
              ⟨_, by rw [mul_comm]; mem_tac⟩, ⟨i, rfl⟩⟩
          letI r : A⁰_ f := HomogeneousLocalization.mk
            ⟨m * i, ⟨proj 𝒜 i b ^ m, by rw [← smul_eq_mul]; mem_tac⟩,
              ⟨_, by rw [mul_comm]; mem_tac⟩, ⟨i, rfl⟩⟩
          l * r
        else
          letI l : A⁰_ f := HomogeneousLocalization.mk
            ⟨m * i, ⟨proj 𝒜 i a ^ m, by rw [← smul_eq_mul]; mem_tac⟩,
              ⟨_, by rw [mul_comm]; mem_tac⟩, ⟨i, rfl⟩⟩
          letI r : A⁰_ f := HomogeneousLocalization.mk
            ⟨m * i, ⟨proj 𝒜 i a ^ (j - m) * proj 𝒜 i b ^ (m + m - j), ?_⟩,
              ⟨_, by rw [mul_comm]; mem_tac⟩, ⟨i, rfl⟩⟩
          l * r
  /-
    case refine_3
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    m : Nat
    f_deg : Membership.mem (𝒜 m) f
    q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
    a b : A
    ha : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrie …
    hb : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrie …
    i : Nat
    g : Nat → HomogeneousLocalization.Away 𝒜 f := fun j => HSMul.hSMul ((HAdd.hAdd …
    ⊢ Membership.mem q.asIdeal (HomogeneousLocalization.mk { deg := HAdd.hAdd (HMu …
  -/
  rotate_left
    /-
      case refine_1
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      f : A
      m : Nat
      f_deg : Membership.mem (𝒜 m) f
      q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
      a b : A
      ha : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrie …
      hb : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrie …
      i j : Nat
      h2 : Not (LT.lt (HAdd.hAdd m m) j)
      h1 : LE.le j m
      ⊢ Membership.mem (𝒜 (HMul.hMul m i)) (HMul.hMul (HPow.hPow ((GradedAlgebra.pro …
    -/
  · rw [(_ : m * i = _)]
    -- Porting note: it seems unification with mul_mem is more fiddly reducing value of mem_tac
    /-
      case refine_1
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      f : A
      m : Nat
      f_deg : Membership.mem (𝒜 m) f
      q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
      a b : A
      ha : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrie …
      hb : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrie …
      i j : Nat
      h2 : Not (LT.lt (HAdd.hAdd m m) j)
      h1 : LE.le j m
      ⊢ Membership.mem (𝒜 ?m.376840) (HMul.hMul (HPow.hPow ((GradedAlgebra.proj 𝒜 i) …
    -/
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/
    apply GradedMonoid.toGradedMul.mul_mem (i := j • i) (j := (m - j) • i) <;> mem_tac_aux
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/
    /-
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      f : A
      m : Nat
      f_deg : Membership.mem (𝒜 m) f
      q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
      a b : A
      ha : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrie …
      hb : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrie …
      i j : Nat
      h2 : Not (LT.lt (HAdd.hAdd m m) j)
      h1 : LE.le j m
      ⊢ Eq (HMul.hMul m i) (HAdd.hAdd (HSMul.hSMul j i) (HSMul.hSMul (HSub.hSub m j) …
    -/
    rw [← add_smul, Nat.add_sub_of_le h1]; rfl
                                           /-
                                             🎉 no goals
                                           -/
    /-
      case refine_2
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      f : A
      m : Nat
      f_deg : Membership.mem (𝒜 m) f
      q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
      a b : A
      ha : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrie …
      hb : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrie …
      i j : Nat
      h2 : Not (LT.lt (HAdd.hAdd m m) j)
      h1 : Not (LE.le j m)
      l : HomogeneousLocalization.Away 𝒜 f := HomogeneousLocalization.mk { deg := HM …
      ⊢ Membership.mem (𝒜 (HMul.hMul m i)) (HMul.hMul (HPow.hPow ((GradedAlgebra.pro …
    -/
  · rw [(_ : m * i = _)]
    /-
      case refine_2
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      f : A
      m : Nat
      f_deg : Membership.mem (𝒜 m) f
      q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
      a b : A
      ha : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrie …
      hb : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrie …
      i j : Nat
      h2 : Not (LT.lt (HAdd.hAdd m m) j)
      h1 : Not (LE.le j m)
      l : HomogeneousLocalization.Away 𝒜 f := HomogeneousLocalization.mk { deg := HM …
      ⊢ Membership.mem (𝒜 ?m.379053) (HMul.hMul (HPow.hPow ((GradedAlgebra.proj 𝒜 i) …
    -/
                                                                                       /-
                                                                                         🎉 no goals
                                                                                       -/
    apply GradedMonoid.toGradedMul.mul_mem (i := (j-m) • i) (j := (m + m - j) • i) <;> mem_tac_aux
                                                                                       /-
                                                                                         🎉 no goals
                                                                                       -/
    /-
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      f : A
      m : Nat
      f_deg : Membership.mem (𝒜 m) f
      q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
      a b : A
      ha : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrie …
      hb : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrie …
      i j : Nat
      h2 : Not (LT.lt (HAdd.hAdd m m) j)
      h1 : Not (LE.le j m)
      l : HomogeneousLocalization.Away 𝒜 f := HomogeneousLocalization.mk { deg := HM …
      ⊢ Eq (HMul.hMul m i) (HAdd.hAdd (HSMul.hSMul (HSub.hSub j m) i) (HSMul.hSMul ( …
    -/
                                                                     /-
                                                                       🎉 no goals
                                                                     -/
    rw [← add_smul]; congr; zify [le_of_not_lt h2, le_of_not_le h1]; abel
                                                                     /-
                                                                       🎉 no goals
                                                                     -/
  /-
    case refine_3
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    m : Nat
    f_deg : Membership.mem (𝒜 m) f
    q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
    a b : A
    ha : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrie …
    hb : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrie …
    i : Nat
    g : Nat → HomogeneousLocalization.Away 𝒜 f := fun j => HSMul.hSMul ((HAdd.hAdd …
    ⊢ Membership.mem q.asIdeal (HomogeneousLocalization.mk { deg := HAdd.hAdd (HMu …
  -/
  convert_to ∑ i ∈ range (m + m + 1), g i ∈ q.1; swap
    /-
      case refine_3
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      f : A
      m : Nat
      f_deg : Membership.mem (𝒜 m) f
      q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
      a b : A
      ha : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrie …
      hb : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrie …
      i : Nat
      g : Nat → HomogeneousLocalization.Away 𝒜 f := fun j => HSMul.hSMul ((HAdd.hAdd …
      ⊢ Membership.mem q.asIdeal ((Finset.range (HAdd.hAdd (HAdd.hAdd m m) 1)).sum f …
    -/
  · refine q.1.sum_mem fun j _ => nsmul_mem ?_ _; split_ifs
    /-
      case pos
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      f : A
      m : Nat
      f_deg : Membership.mem (𝒜 m) f
      q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
      a b : A
      ha : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrie …
      hb : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrie …
      i : Nat
      g : Nat → HomogeneousLocalization.Away 𝒜 f := fun j => HSMul.hSMul ((HAdd.hAdd …
      j : Nat
      x✝ : Membership.mem (Finset.range (HAdd.hAdd (HAdd.hAdd m m) 1)) j
      h✝ : LT.lt (HAdd.hAdd m m) j
      ⊢ Membership.mem q.asIdeal 0
    -/
    exacts [q.1.zero_mem, q.1.mul_mem_left _ (hb i), q.1.mul_mem_right _ (ha i)]
    /-
      🎉 no goals
    -/
  /-
    case h.e'_5
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    m : Nat
    f_deg : Membership.mem (𝒜 m) f
    q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
    a b : A
    ha : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrie …
    hb : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrie …
    i : Nat
    g : Nat → HomogeneousLocalization.Away 𝒜 f := fun j => HSMul.hSMul ((HAdd.hAdd …
    ⊢ Eq (HomogeneousLocalization.mk { deg := HAdd.hAdd (HMul.hMul m i) (HMul.hMul …
  -/
  rw [HomogeneousLocalization.ext_iff_val, HomogeneousLocalization.val_mk]
  /-
    case h.e'_5
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    m : Nat
    f_deg : Membership.mem (𝒜 m) f
    q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
    a b : A
    ha : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrie …
    hb : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrie …
    i : Nat
    g : Nat → HomogeneousLocalization.Away 𝒜 f := fun j => HSMul.hSMul ((HAdd.hAdd …
    ⊢ Eq (Localization.mk ↑{ deg := HAdd.hAdd (HMul.hMul m i) (HMul.hMul m i), num …
  -/
  change _ = (algebraMap (HomogeneousLocalization.Away 𝒜 f) (Localization.Away f)) _
  /-
    case h.e'_5
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    m : Nat
    f_deg : Membership.mem (𝒜 m) f
    q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
    a b : A
    ha : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrie …
    hb : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrie …
    i : Nat
    g : Nat → HomogeneousLocalization.Away 𝒜 f := fun j => HSMul.hSMul ((HAdd.hAdd …
    ⊢ Eq (Localization.mk ↑{ deg := HAdd.hAdd (HMul.hMul m i) (HMul.hMul m i), num …
  -/
  dsimp only [Subtype.coe_mk]; rw [map_sum, mk_sum]
  /-
    case h.e'_5
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    m : Nat
    f_deg : Membership.mem (𝒜 m) f
    q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
    a b : A
    ha : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrie …
    hb : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrie …
    i : Nat
    g : Nat → HomogeneousLocalization.Away 𝒜 f := fun j => HSMul.hSMul ((HAdd.hAdd …
    ⊢ Eq ((Finset.range (HAdd.hAdd (HAdd.hAdd m m) 1)).sum fun i_1 => Localization …
  -/
  apply Finset.sum_congr rfl fun j hj => _
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    m : Nat
    f_deg : Membership.mem (𝒜 m) f
    q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
    a b : A
    ha : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrie …
    hb : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrie …
    i : Nat
    g : Nat → HomogeneousLocalization.Away 𝒜 f := fun j => HSMul.hSMul ((HAdd.hAdd …
    ⊢ ∀ (j : Nat), Membership.mem (Finset.range (HAdd.hAdd (HAdd.hAdd m m) 1)) j → …
  -/
  intro j hj
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    m : Nat
    f_deg : Membership.mem (𝒜 m) f
    q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
    a b : A
    ha : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrie …
    hb : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrie …
    i : Nat
    g : Nat → HomogeneousLocalization.Away 𝒜 f := fun j => HSMul.hSMul ((HAdd.hAdd …
    j : Nat
    hj : Membership.mem (Finset.range (HAdd.hAdd (HAdd.hAdd m m) 1)) j
    ⊢ Eq (Localization.mk (HSMul.hSMul ((HAdd.hAdd m m).choose j) (HMul.hMul (HPow …
  -/
  change _ = HomogeneousLocalization.val _
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    m : Nat
    f_deg : Membership.mem (𝒜 m) f
    q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
    a b : A
    ha : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrie …
    hb : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrie …
    i : Nat
    g : Nat → HomogeneousLocalization.Away 𝒜 f := fun j => HSMul.hSMul ((HAdd.hAdd …
    j : Nat
    hj : Membership.mem (Finset.range (HAdd.hAdd (HAdd.hAdd m m) 1)) j
    ⊢ Eq (Localization.mk (HSMul.hSMul ((HAdd.hAdd m m).choose j) (HMul.hMul (HPow …
  -/
  rw [HomogeneousLocalization.val_smul]
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    m : Nat
    f_deg : Membership.mem (𝒜 m) f
    q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
    a b : A
    ha : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrie …
    hb : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrie …
    i : Nat
    g : Nat → HomogeneousLocalization.Away 𝒜 f := fun j => HSMul.hSMul ((HAdd.hAdd …
    j : Nat
    hj : Membership.mem (Finset.range (HAdd.hAdd (HAdd.hAdd m m) 1)) j
    ⊢ Eq (Localization.mk (HSMul.hSMul ((HAdd.hAdd m m).choose j) (HMul.hMul (HPow …
  -/
  split_ifs with h2 h1
    /-
      case pos
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      f : A
      m : Nat
      f_deg : Membership.mem (𝒜 m) f
      q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
      a b : A
      ha : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrie …
      hb : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrie …
      i : Nat
      g : Nat → HomogeneousLocalization.Away 𝒜 f := fun j => HSMul.hSMul ((HAdd.hAdd …
      j : Nat
      hj : Membership.mem (Finset.range (HAdd.hAdd (HAdd.hAdd m m) 1)) j
      h2 : LT.lt (HAdd.hAdd m m) j
      ⊢ Eq (Localization.mk (HSMul.hSMul ((HAdd.hAdd m m).choose j) (HMul.hMul (HPow …
    -/
  · exact ((Finset.mem_range.1 hj).not_le h2).elim
    /-
      🎉 no goals
    -/
  all_goals simp only [HomogeneousLocalization.val_mul, HomogeneousLocalization.val_zero,
    HomogeneousLocalization.val_mk, Subtype.coe_mk, Localization.mk_mul, ← smul_mk]; congr 2
    /-
      case pos.e_a.e_x
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      f : A
      m : Nat
      f_deg : Membership.mem (𝒜 m) f
      q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
      a b : A
      ha : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrie …
      hb : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrie …
      i : Nat
      g : Nat → HomogeneousLocalization.Away 𝒜 f := fun j => HSMul.hSMul ((HAdd.hAdd …
      j : Nat
      hj : Membership.mem (Finset.range (HAdd.hAdd (HAdd.hAdd m m) 1)) j
      h2 : Not (LT.lt (HAdd.hAdd m m) j)
      h1 : LE.le j m
      ⊢ Eq (HMul.hMul (HPow.hPow ((GradedAlgebra.proj 𝒜 i) a) j) (HPow.hPow ((Graded …
    -/
  · dsimp; rw [mul_assoc, ← pow_add, add_comm (m - j), Nat.add_sub_assoc h1]
           /-
             🎉 no goals
           -/
    /-
      case pos.e_a.e_y
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      f : A
      m : Nat
      f_deg : Membership.mem (𝒜 m) f
      q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
      a b : A
      ha : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrie …
      hb : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrie …
      i : Nat
      g : Nat → HomogeneousLocalization.Away 𝒜 f := fun j => HSMul.hSMul ((HAdd.hAdd …
      j : Nat
      hj : Membership.mem (Finset.range (HAdd.hAdd (HAdd.hAdd m m) 1)) j
      h2 : Not (LT.lt (HAdd.hAdd m m) j)
      h1 : LE.le j m
      ⊢ Eq ⟨HPow.hPow f (HAdd.hAdd i i), ⋯⟩ (HMul.hMul ⟨HPow.hPow f i, ⋯⟩ ⟨HPow.hPow …
    -/
  · simp_rw [pow_add]; rfl
                       /-
                         🎉 no goals
                       -/
    /-
      case neg.e_a.e_x
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      f : A
      m : Nat
      f_deg : Membership.mem (𝒜 m) f
      q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
      a b : A
      ha : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrie …
      hb : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrie …
      i : Nat
      g : Nat → HomogeneousLocalization.Away 𝒜 f := fun j => HSMul.hSMul ((HAdd.hAdd …
      j : Nat
      hj : Membership.mem (Finset.range (HAdd.hAdd (HAdd.hAdd m m) 1)) j
      h2 : Not (LT.lt (HAdd.hAdd m m) j)
      h1 : Not (LE.le j m)
      ⊢ Eq (HMul.hMul (HPow.hPow ((GradedAlgebra.proj 𝒜 i) a) j) (HPow.hPow ((Graded …
    -/
  · dsimp; rw [← mul_assoc, ← pow_add, Nat.add_sub_of_le (le_of_not_le h1)]
           /-
             🎉 no goals
           -/
    /-
      case neg.e_a.e_y
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      f : A
      m : Nat
      f_deg : Membership.mem (𝒜 m) f
      q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
      a b : A
      ha : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrie …
      hb : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrie …
      i : Nat
      g : Nat → HomogeneousLocalization.Away 𝒜 f := fun j => HSMul.hSMul ((HAdd.hAdd …
      j : Nat
      hj : Membership.mem (Finset.range (HAdd.hAdd (HAdd.hAdd m m) 1)) j
      h2 : Not (LT.lt (HAdd.hAdd m m) j)
      h1 : Not (LE.le j m)
      ⊢ Eq ⟨HPow.hPow f (HAdd.hAdd i i), ⋯⟩ (HMul.hMul ⟨HPow.hPow f i, ⋯⟩ ⟨HPow.hPow …
    -/
  · simp_rw [pow_add]; rfl
                       /-
                         🎉 no goals
                       -/


theorem carrier.zero_mem : (0 : A) ∈ carrier f_deg q := fun i => by
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    m : Nat
    f_deg : Membership.mem (𝒜 m) f
    hm : LT.lt 0 m
    q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
    i : Nat
    ⊢ Membership.mem q.asIdeal (HomogeneousLocalization.mk { deg := HMul.hMul m i, …
  -/
  convert Submodule.zero_mem q.1 using 1
  rw [HomogeneousLocalization.ext_iff_val, HomogeneousLocalization.val_mk,
                                       /-
                                         case h.e'_5
                                         R : Type u_1
                                         A : Type u_2
                                         inst✝³ : CommRing R
                                         inst✝² : CommRing A
                                         inst✝¹ : Algebra R A
                                         𝒜 : Nat → Submodule R A
                                         inst✝ : GradedAlgebra 𝒜
                                         f : A
                                         m : Nat
                                         f_deg : Membership.mem (𝒜 m) f
                                         hm : LT.lt 0 m
                                         q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
                                         i : Nat
                                         ⊢ Eq (Localization.mk ↑{ deg := HMul.hMul m i, num := ⟨HPow.hPow ((GradedAlgeb …
                                       -/
    HomogeneousLocalization.val_zero]; simp_rw [map_zero, zero_pow hm.ne']
  /-
    case h.e'_5
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    m : Nat
    f_deg : Membership.mem (𝒜 m) f
    hm : LT.lt 0 m
    q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
    i : Nat
    ⊢ Eq (Localization.mk 0 ⟨HPow.hPow f i, ⋯⟩) 0
  -/
  convert Localization.mk_zero (S := Submonoid.powers f) _ using 1
  /-
    🎉 no goals
  -/


theorem carrier.smul_mem (c x : A) (hx : x ∈ carrier f_deg q) : c • x ∈ carrier f_deg q := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    m : Nat
    f_deg : Membership.mem (𝒜 m) f
    hm : LT.lt 0 m
    q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
    c x : A
    hx : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrie …
    ⊢ Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrier f …
  -/
  revert c
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    m : Nat
    f_deg : Membership.mem (𝒜 m) f
    hm : LT.lt 0 m
    q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
    x : A
    hx : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrie …
    ⊢ ∀ (c : A), Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpe …
  -/
  refine DirectSum.Decomposition.inductionOn 𝒜 ?_ ?_ ?_
    /-
      case refine_1
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      f : A
      m : Nat
      f_deg : Membership.mem (𝒜 m) f
      hm : LT.lt 0 m
      q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
      x : A
      hx : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrie …
      ⊢ Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrier f …
    -/
  · rw [zero_smul]; exact carrier.zero_mem f_deg hm _
                    /-
                      🎉 no goals
                    -/
    /-
      case refine_2
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      f : A
      m : Nat
      f_deg : Membership.mem (𝒜 m) f
      hm : LT.lt 0 m
      q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
      x : A
      hx : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrie …
      ⊢ ∀ {i : Nat} (m_1 : Subtype fun x => Membership.mem (𝒜 i) x), Membership.mem  …
    -/
  · rintro n ⟨a, ha⟩ i
    /-
      case refine_2.mk
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      f : A
      m : Nat
      f_deg : Membership.mem (𝒜 m) f
      hm : LT.lt 0 m
      q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
      x : A
      hx : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrie …
      n : Nat
      a : A
      ha : Membership.mem (𝒜 n) a
      i : Nat
      ⊢ Membership.mem q.asIdeal (HomogeneousLocalization.mk { deg := HMul.hMul m i, …
    -/
    simp_rw [proj_apply, smul_eq_mul, coe_decompose_mul_of_left_mem 𝒜 i ha]
    -- Porting note: having trouble with Mul instance
    let product : A⁰_ f :=
      Mul.mul (HomogeneousLocalization.mk ⟨_, ⟨a ^ m, pow_mem_graded m ha⟩, ⟨_, ?_⟩, ⟨n, rfl⟩⟩)
        (HomogeneousLocalization.mk ⟨_, ⟨proj 𝒜 (i - n) x ^ m, by mem_tac⟩, ⟨_, ?_⟩, ⟨i - n, rfl⟩⟩)
      /-
        case refine_2.mk.refine_3
        R : Type u_1
        A : Type u_2
        inst✝³ : CommRing R
        inst✝² : CommRing A
        inst✝¹ : Algebra R A
        𝒜 : Nat → Submodule R A
        inst✝ : GradedAlgebra 𝒜
        f : A
        m : Nat
        f_deg : Membership.mem (𝒜 m) f
        hm : LT.lt 0 m
        q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
        x : A
        hx : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrie …
        n : Nat
        a : A
        ha : Membership.mem (𝒜 n) a
        i : Nat
        product : HomogeneousLocalization.Away 𝒜 f := Mul.mul (HomogeneousLocalization …
        ⊢ Membership.mem q.asIdeal (HomogeneousLocalization.mk { deg := HMul.hMul m i, …
      -/
    · split_ifs with h
        /-
          case pos
          R : Type u_1
          A : Type u_2
          inst✝³ : CommRing R
          inst✝² : CommRing A
          inst✝¹ : Algebra R A
          𝒜 : Nat → Submodule R A
          inst✝ : GradedAlgebra 𝒜
          f : A
          m : Nat
          f_deg : Membership.mem (𝒜 m) f
          hm : LT.lt 0 m
          q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
          x : A
          hx : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrie …
          n : Nat
          a : A
          ha : Membership.mem (𝒜 n) a
          i : Nat
          product : HomogeneousLocalization.Away 𝒜 f := Mul.mul (HomogeneousLocalization …
          h : LE.le n i
          ⊢ Membership.mem q.asIdeal (HomogeneousLocalization.mk { deg := HMul.hMul m i, …
        -/
      · convert_to product ∈ q.1
          /-
            case h.e'_5
            R : Type u_1
            A : Type u_2
            inst✝³ : CommRing R
            inst✝² : CommRing A
            inst✝¹ : Algebra R A
            𝒜 : Nat → Submodule R A
            inst✝ : GradedAlgebra 𝒜
            f : A
            m : Nat
            f_deg : Membership.mem (𝒜 m) f
            hm : LT.lt 0 m
            q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
            x : A
            hx : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrie …
            n : Nat
            a : A
            ha : Membership.mem (𝒜 n) a
            i : Nat
            product : HomogeneousLocalization.Away 𝒜 f := Mul.mul (HomogeneousLocalization …
            h : LE.le n i
            ⊢ Eq (HomogeneousLocalization.mk { deg := HMul.hMul m i, num := ⟨HPow.hPow (HM …
          -/
        · dsimp [product]
          erw [HomogeneousLocalization.ext_iff_val, HomogeneousLocalization.val_mk,
            HomogeneousLocalization.val_mul, HomogeneousLocalization.val_mk,
            HomogeneousLocalization.val_mk]
            /-
              case h.e'_5
              R : Type u_1
              A : Type u_2
              inst✝³ : CommRing R
              inst✝² : CommRing A
              inst✝¹ : Algebra R A
              𝒜 : Nat → Submodule R A
              inst✝ : GradedAlgebra 𝒜
              f : A
              m : Nat
              f_deg : Membership.mem (𝒜 m) f
              hm : LT.lt 0 m
              q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
              x : A
              hx : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrie …
              n : Nat
              a : A
              ha : Membership.mem (𝒜 n) a
              i : Nat
              product : HomogeneousLocalization.Away 𝒜 f := Mul.mul (HomogeneousLocalization …
              h : LE.le n i
              ⊢ Eq (Localization.mk ↑{ deg := HMul.hMul m i, num := ⟨HPow.hPow (HMul.hMul a  …
            -/
          · simp_rw [mul_pow]; rw [Localization.mk_mul]
              /-
                case h.e'_5
                R : Type u_1
                A : Type u_2
                inst✝³ : CommRing R
                inst✝² : CommRing A
                inst✝¹ : Algebra R A
                𝒜 : Nat → Submodule R A
                inst✝ : GradedAlgebra 𝒜
                f : A
                m : Nat
                f_deg : Membership.mem (𝒜 m) f
                hm : LT.lt 0 m
                q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
                x : A
                hx : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrie …
                n : Nat
                a : A
                ha : Membership.mem (𝒜 n) a
                i : Nat
                product : HomogeneousLocalization.Away 𝒜 f := Mul.mul (HomogeneousLocalization …
                h : LE.le n i
                ⊢ Eq (Localization.mk (HMul.hMul (HPow.hPow a m) (HPow.hPow (↑(((DirectSum.dec …
              -/
            · congr; rw [← pow_add, Nat.add_sub_of_le h]
                     /-
                       🎉 no goals
                     -/
          /-
            case pos
            R : Type u_1
            A : Type u_2
            inst✝³ : CommRing R
            inst✝² : CommRing A
            inst✝¹ : Algebra R A
            𝒜 : Nat → Submodule R A
            inst✝ : GradedAlgebra 𝒜
            f : A
            m : Nat
            f_deg : Membership.mem (𝒜 m) f
            hm : LT.lt 0 m
            q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
            x : A
            hx : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrie …
            n : Nat
            a : A
            ha : Membership.mem (𝒜 n) a
            i : Nat
            product : HomogeneousLocalization.Away 𝒜 f := Mul.mul (HomogeneousLocalization …
            h : LE.le n i
            ⊢ Membership.mem q.asIdeal product
          -/
        · apply Ideal.mul_mem_left (α := A⁰_ f) _ _ (hx _)
          /-
            case refine_2.mk.refine_1
            R : Type u_1
            A : Type u_2
            inst✝³ : CommRing R
            inst✝² : CommRing A
            inst✝¹ : Algebra R A
            𝒜 : Nat → Submodule R A
            inst✝ : GradedAlgebra 𝒜
            f : A
            m : Nat
            f_deg : Membership.mem (𝒜 m) f
            hm : LT.lt 0 m
            q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
            x : A
            hx : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrie …
            n : Nat
            a : A
            ha : Membership.mem (𝒜 n) a
            i : Nat
            ⊢ Membership.mem (𝒜 (HSMul.hSMul m n)) (HPow.hPow f n)
          -/
          rw [(_ : m • n = _)]
            /-
              case refine_2.mk.refine_1
              R : Type u_1
              A : Type u_2
              inst✝³ : CommRing R
              inst✝² : CommRing A
              inst✝¹ : Algebra R A
              𝒜 : Nat → Submodule R A
              inst✝ : GradedAlgebra 𝒜
              f : A
              m : Nat
              f_deg : Membership.mem (𝒜 m) f
              hm : LT.lt 0 m
              q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
              x : A
              hx : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrie …
              n : Nat
              a : A
              ha : Membership.mem (𝒜 n) a
              i : Nat
              ⊢ Membership.mem (𝒜 ?m.467268) (HPow.hPow f n)
            -/
          · mem_tac
            /-
              🎉 no goals
            -/
            /-
              R : Type u_1
              A : Type u_2
              inst✝³ : CommRing R
              inst✝² : CommRing A
              inst✝¹ : Algebra R A
              𝒜 : Nat → Submodule R A
              inst✝ : GradedAlgebra 𝒜
              f : A
              m : Nat
              f_deg : Membership.mem (𝒜 m) f
              hm : LT.lt 0 m
              q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
              x : A
              hx : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrie …
              n : Nat
              a : A
              ha : Membership.mem (𝒜 n) a
              i : Nat
              ⊢ Eq (HSMul.hSMul m n) (HSMul.hSMul n m)
            -/
          · simp only [smul_eq_mul, mul_comm]
            /-
              🎉 no goals
            -/
        /-
          case neg
          R : Type u_1
          A : Type u_2
          inst✝³ : CommRing R
          inst✝² : CommRing A
          inst✝¹ : Algebra R A
          𝒜 : Nat → Submodule R A
          inst✝ : GradedAlgebra 𝒜
          f : A
          m : Nat
          f_deg : Membership.mem (𝒜 m) f
          hm : LT.lt 0 m
          q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
          x : A
          hx : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrie …
          n : Nat
          a : A
          ha : Membership.mem (𝒜 n) a
          i : Nat
          product : HomogeneousLocalization.Away 𝒜 f := Mul.mul (HomogeneousLocalization …
          h : Not (LE.le n i)
          ⊢ Membership.mem q.asIdeal (HomogeneousLocalization.mk { deg := HMul.hMul m i, …
        -/
      · simpa only [map_zero, zero_pow hm.ne'] using zero_mem f_deg hm q i
        /-
          🎉 no goals
        -/
    /-
      case refine_2.mk.refine_2
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      f : A
      m : Nat
      f_deg : Membership.mem (𝒜 m) f
      hm : LT.lt 0 m
      q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
      x : A
      hx : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrie …
      n : Nat
      a : A
      ha : Membership.mem (𝒜 n) a
      i : Nat
      ⊢ Membership.mem (𝒜 (HSMul.hSMul m (HSub.hSub i n))) (HPow.hPow f (HSub.hSub i …
    -/
    rw [(_ : m • (i - n) = _)]
      /-
        case refine_2.mk.refine_2
        R : Type u_1
        A : Type u_2
        inst✝³ : CommRing R
        inst✝² : CommRing A
        inst✝¹ : Algebra R A
        𝒜 : Nat → Submodule R A
        inst✝ : GradedAlgebra 𝒜
        f : A
        m : Nat
        f_deg : Membership.mem (𝒜 m) f
        hm : LT.lt 0 m
        q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
        x : A
        hx : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrie …
        n : Nat
        a : A
        ha : Membership.mem (𝒜 n) a
        i : Nat
        ⊢ Membership.mem (𝒜 ?m.475074) (HPow.hPow f (HSub.hSub i n))
      -/
    · mem_tac
      /-
        🎉 no goals
      -/
      /-
        R : Type u_1
        A : Type u_2
        inst✝³ : CommRing R
        inst✝² : CommRing A
        inst✝¹ : Algebra R A
        𝒜 : Nat → Submodule R A
        inst✝ : GradedAlgebra 𝒜
        f : A
        m : Nat
        f_deg : Membership.mem (𝒜 m) f
        hm : LT.lt 0 m
        q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
        x : A
        hx : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrie …
        n : Nat
        a : A
        ha : Membership.mem (𝒜 n) a
        i : Nat
        ⊢ Eq (HSMul.hSMul m (HSub.hSub i n)) (HSMul.hSMul (HSub.hSub i n) m)
      -/
    · simp only [smul_eq_mul, mul_comm]
      /-
        🎉 no goals
      -/
    /-
      case refine_3
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      f : A
      m : Nat
      f_deg : Membership.mem (𝒜 m) f
      hm : LT.lt 0 m
      q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
      x : A
      hx : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrie …
      ⊢ ∀ (m_1 m' : A), Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.Fr …
    -/
  · simp_rw [add_smul]; exact fun _ _ => carrier.add_mem f_deg q
                        /-
                          🎉 no goals
                        -/


/-- For a prime ideal `q` in `A⁰_f`, the set `{a | aᵢᵐ/fⁱ ∈ q}` as an ideal.
-/
def carrier.asIdeal : Ideal A where
  carrier := carrier f_deg q
  zero_mem' := carrier.zero_mem f_deg hm q
  add_mem' := carrier.add_mem f_deg q
  smul_mem' := carrier.smul_mem f_deg hm q



theorem carrier.asIdeal.homogeneous : (carrier.asIdeal f_deg hm q).IsHomogeneous 𝒜 :=
  fun i a ha j =>
                                     /-
                                       R : Type u_1
                                       A : Type u_2
                                       inst✝³ : CommRing R
                                       inst✝² : CommRing A
                                       inst✝¹ : Algebra R A
                                       𝒜 : Nat → Submodule R A
                                       inst✝ : GradedAlgebra 𝒜
                                       f : A
                                       m : Nat
                                       f_deg : Membership.mem (𝒜 m) f
                                       hm : LT.lt 0 m
                                       q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
                                       i : Nat
                                       a : A
                                       ha : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carrie …
                                       j : Nat
                                       h : Eq i j
                                       ⊢ Membership.mem q.asIdeal (HomogeneousLocalization.mk { deg := HMul.hMul m i, …
                                     -/
  (em (i = j)).elim (fun h => h ▸ by simpa only [proj_apply, decompose_coe, of_eq_same] using ha _)
                                     /-
                                       🎉 no goals
                                     -/
    fun h => by
    simpa only [proj_apply, decompose_of_mem_ne 𝒜 (Submodule.coe_mem (decompose 𝒜 a i)) h,
      zero_pow hm.ne', map_zero] using carrier.zero_mem f_deg hm q j


/-- For a prime ideal `q` in `A⁰_f`, the set `{a | aᵢᵐ/fⁱ ∈ q}` as a homogeneous ideal.
-/
def carrier.asHomogeneousIdeal : HomogeneousIdeal 𝒜 :=
  ⟨carrier.asIdeal f_deg hm q, carrier.asIdeal.homogeneous f_deg hm q⟩


theorem carrier.denom_not_mem : f ∉ carrier.asIdeal f_deg hm q := fun rid =>
  q.isPrime.ne_top <|
    (Ideal.eq_top_iff_one _).mpr
      (by
        /-
          R : Type u_1
          A : Type u_2
          inst✝³ : CommRing R
          inst✝² : CommRing A
          inst✝¹ : Algebra R A
          𝒜 : Nat → Submodule R A
          inst✝ : GradedAlgebra 𝒜
          f : A
          m : Nat
          f_deg : Membership.mem (𝒜 m) f
          hm : LT.lt 0 m
          q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
          rid : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carri …
          ⊢ Membership.mem q.asIdeal 1
        -/
        convert rid m
        rw [HomogeneousLocalization.ext_iff_val, HomogeneousLocalization.val_one,
          HomogeneousLocalization.val_mk]
        /-
          case h.e'_5
          R : Type u_1
          A : Type u_2
          inst✝³ : CommRing R
          inst✝² : CommRing A
          inst✝¹ : Algebra R A
          𝒜 : Nat → Submodule R A
          inst✝ : GradedAlgebra 𝒜
          f : A
          m : Nat
          f_deg : Membership.mem (𝒜 m) f
          hm : LT.lt 0 m
          q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
          rid : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carri …
          ⊢ Eq 1 (Localization.mk ↑{ deg := HMul.hMul m m, num := ⟨HPow.hPow ((GradedAlg …
        -/
        dsimp
        /-
          case h.e'_5
          R : Type u_1
          A : Type u_2
          inst✝³ : CommRing R
          inst✝² : CommRing A
          inst✝¹ : Algebra R A
          𝒜 : Nat → Submodule R A
          inst✝ : GradedAlgebra 𝒜
          f : A
          m : Nat
          f_deg : Membership.mem (𝒜 m) f
          hm : LT.lt 0 m
          q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
          rid : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carri …
          ⊢ Eq 1 (Localization.mk (HPow.hPow (↑(((DirectSum.decompose 𝒜) f) m)) m) ⟨HPow …
        -/
        simp_rw [decompose_of_mem_same _ f_deg]
        /-
          case h.e'_5
          R : Type u_1
          A : Type u_2
          inst✝³ : CommRing R
          inst✝² : CommRing A
          inst✝¹ : Algebra R A
          𝒜 : Nat → Submodule R A
          inst✝ : GradedAlgebra 𝒜
          f : A
          m : Nat
          f_deg : Membership.mem (𝒜 m) f
          hm : LT.lt 0 m
          q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
          rid : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carri …
          ⊢ Eq 1 (Localization.mk (HPow.hPow f m) ⟨HPow.hPow f m, ⋯⟩)
        -/
        simp only [mk_eq_monoidOf_mk', Submonoid.LocalizationMap.mk'_self])
        /-
          🎉 no goals
        -/


theorem carrier.relevant : ¬HomogeneousIdeal.irrelevant 𝒜 ≤ carrier.asHomogeneousIdeal f_deg hm q :=
  fun rid => carrier.denom_not_mem f_deg hm q <| rid <| DirectSum.decompose_of_mem_ne 𝒜 f_deg hm.ne'


theorem carrier.asIdeal.ne_top : carrier.asIdeal f_deg hm q ≠ ⊤ := fun rid =>
  carrier.denom_not_mem f_deg hm q (rid.symm ▸ Submodule.mem_top)


theorem carrier.asIdeal.prime : (carrier.asIdeal f_deg hm q).IsPrime :=
  (carrier.asIdeal.homogeneous f_deg hm q).isPrime_of_homogeneous_mem_or_mem
    (carrier.asIdeal.ne_top f_deg hm q) fun {x y} ⟨nx, hnx⟩ ⟨ny, hny⟩ hxy =>
    show (∀ _, _ ∈ _) ∨ ∀ _, _ ∈ _ by
      /-
        R : Type u_1
        A : Type u_2
        inst✝³ : CommRing R
        inst✝² : CommRing A
        inst✝¹ : Algebra R A
        𝒜 : Nat → Submodule R A
        inst✝ : GradedAlgebra 𝒜
        f : A
        m : Nat
        f_deg : Membership.mem (𝒜 m) f
        hm : LT.lt 0 m
        q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
        x y : A
        x✝¹ : SetLike.Homogeneous 𝒜 x
        x✝ : SetLike.Homogeneous 𝒜 y
        hxy : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carri …
        nx : Nat
        hnx : Membership.mem (𝒜 nx) x
        ny : Nat
        hny : Membership.mem (𝒜 ny) y
        ⊢ Or (∀ (x_1 : Nat), Membership.mem q.asIdeal (HomogeneousLocalization.mk { de …
      -/
      rw [← and_forall_ne nx, and_iff_left, ← and_forall_ne ny, and_iff_left]
        /-
          R : Type u_1
          A : Type u_2
          inst✝³ : CommRing R
          inst✝² : CommRing A
          inst✝¹ : Algebra R A
          𝒜 : Nat → Submodule R A
          inst✝ : GradedAlgebra 𝒜
          f : A
          m : Nat
          f_deg : Membership.mem (𝒜 m) f
          hm : LT.lt 0 m
          q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
          x y : A
          x✝¹ : SetLike.Homogeneous 𝒜 x
          x✝ : SetLike.Homogeneous 𝒜 y
          hxy : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carri …
          nx : Nat
          hnx : Membership.mem (𝒜 nx) x
          ny : Nat
          hny : Membership.mem (𝒜 ny) y
          ⊢ Or (Membership.mem q.asIdeal (HomogeneousLocalization.mk { deg := HMul.hMul  …
        -/
      · apply q.2.mem_or_mem; convert hxy (nx + ny) using 1
        /-
          case h.e'_5
          R : Type u_1
          A : Type u_2
          inst✝³ : CommRing R
          inst✝² : CommRing A
          inst✝¹ : Algebra R A
          𝒜 : Nat → Submodule R A
          inst✝ : GradedAlgebra 𝒜
          f : A
          m : Nat
          f_deg : Membership.mem (𝒜 m) f
          hm : LT.lt 0 m
          q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
          x y : A
          x✝¹ : SetLike.Homogeneous 𝒜 x
          x✝ : SetLike.Homogeneous 𝒜 y
          hxy : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carri …
          nx : Nat
          hnx : Membership.mem (𝒜 nx) x
          ny : Nat
          hny : Membership.mem (𝒜 ny) y
          ⊢ Eq (HMul.hMul (HomogeneousLocalization.mk { deg := HMul.hMul m nx, num := ⟨H …
        -/
        dsimp
        simp_rw [decompose_of_mem_same 𝒜 hnx, decompose_of_mem_same 𝒜 hny,
          decompose_of_mem_same 𝒜 (SetLike.GradedMonoid.toGradedMul.mul_mem hnx hny),
          mul_pow, pow_add]
        simp only [HomogeneousLocalization.ext_iff_val, HomogeneousLocalization.val_mk,
          HomogeneousLocalization.val_mul, Localization.mk_mul]
        /-
          case h.e'_5
          R : Type u_1
          A : Type u_2
          inst✝³ : CommRing R
          inst✝² : CommRing A
          inst✝¹ : Algebra R A
          𝒜 : Nat → Submodule R A
          inst✝ : GradedAlgebra 𝒜
          f : A
          m : Nat
          f_deg : Membership.mem (𝒜 m) f
          hm : LT.lt 0 m
          q : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
          x y : A
          x✝¹ : SetLike.Homogeneous 𝒜 x
          x✝ : SetLike.Homogeneous 𝒜 y
          hxy : Membership.mem (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.carri …
          nx : Nat
          hnx : Membership.mem (𝒜 nx) x
          ny : Nat
          hny : Membership.mem (𝒜 ny) y
          ⊢ Eq (Localization.mk (HMul.hMul (HPow.hPow x m) (HPow.hPow y m)) (HMul.hMul ⟨ …
        -/
        simp only [Submonoid.mk_mul_mk, mk_eq_monoidOf_mk']
        /-
          🎉 no goals
        -/
      all_goals
        intro n hn; convert q.1.zero_mem using 1
        rw [HomogeneousLocalization.ext_iff_val, HomogeneousLocalization.val_mk,
          HomogeneousLocalization.val_zero]; simp_rw [proj_apply]
        convert mk_zero (S := Submonoid.powers f) _
        rw [decompose_of_mem_ne 𝒜 _ hn.symm, zero_pow hm.ne']
        · first | exact hnx | exact hny


/-- The function `Spec A⁰_f → Proj|D(f)` sending `q` to `{a | aᵢᵐ/fⁱ ∈ q}`. -/
def toFun : (Spec.T A⁰_ f) → Proj.T| pbo f := fun q =>
  ⟨⟨carrier.asHomogeneousIdeal f_deg hm q, carrier.asIdeal.prime f_deg hm q,
      carrier.relevant f_deg hm q⟩,
    (ProjectiveSpectrum.mem_basicOpen _ f _).mp <| carrier.denom_not_mem f_deg hm q⟩


lemma toSpec_fromSpec {f : A} {m : ℕ} (f_deg : f ∈ 𝒜 m) (hm : 0 < m) (x : Spec.T (A⁰_ f)) :
    toSpec 𝒜 f (FromSpec.toFun f_deg hm x) = x := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    m : Nat
    f_deg : Membership.mem (𝒜 m) f
    hm : LT.lt 0 m
    x : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
    ⊢ Eq ((AlgebraicGeometry.ProjIsoSpecTopComponent.toSpec 𝒜 f) (AlgebraicGeometr …
  -/
  apply PrimeSpectrum.ext
  /-
    case asIdeal
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    m : Nat
    f_deg : Membership.mem (𝒜 m) f
    hm : LT.lt 0 m
    x : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
    ⊢ Eq ((AlgebraicGeometry.ProjIsoSpecTopComponent.toSpec 𝒜 f) (AlgebraicGeometr …
  -/
  ext z
  /-
    case asIdeal.h
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    m : Nat
    f_deg : Membership.mem (𝒜 m) f
    hm : LT.lt 0 m
    x : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
    z : ↑(CommRingCat.of (HomogeneousLocalization.Away 𝒜 f))
    ⊢ Iff (Membership.mem ((AlgebraicGeometry.ProjIsoSpecTopComponent.toSpec 𝒜 f)  …
  -/
  obtain ⟨z, rfl⟩ := HomogeneousLocalization.mk_surjective z
  /-
    case asIdeal.h.intro
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    m : Nat
    f_deg : Membership.mem (𝒜 m) f
    hm : LT.lt 0 m
    x : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
    z : HomogeneousLocalization.NumDenSameDeg 𝒜 (Submonoid.powers f)
    ⊢ Iff (Membership.mem ((AlgebraicGeometry.ProjIsoSpecTopComponent.toSpec 𝒜 f)  …
  -/
  rw [← FromSpec.num_mem_carrier_iff f_deg hm x]
  /-
    case asIdeal.h.intro
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    m : Nat
    f_deg : Membership.mem (𝒜 m) f
    hm : LT.lt 0 m
    x : ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Homogeneo …
    z : HomogeneousLocalization.NumDenSameDeg 𝒜 (Submonoid.powers f)
    ⊢ Iff (Membership.mem ((AlgebraicGeometry.ProjIsoSpecTopComponent.toSpec 𝒜 f)  …
  -/
  exact ToSpec.mk_mem_carrier _ z
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-03-02")] alias toSpecFromSpec := toSpec_fromSpec


lemma fromSpec_toSpec {f : A} {m : ℕ} (f_deg : f ∈ 𝒜 m) (hm : 0 < m) (x : Proj.T| pbo f) :
    FromSpec.toFun f_deg hm (toSpec 𝒜 f x) = x := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    m : Nat
    f_deg : Membership.mem (𝒜 m) f
    hm : LT.lt 0 m
    x : ↑↑((AlgebraicGeometry.Proj.toLocallyRingedSpace 𝒜).restrict ⋯).toPresheafe …
    ⊢ Eq (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.toFun f_deg hm ((Alge …
  -/
  refine Subtype.ext <| ProjectiveSpectrum.ext <| HomogeneousIdeal.ext' ?_
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    m : Nat
    f_deg : Membership.mem (𝒜 m) f
    hm : LT.lt 0 m
    x : ↑↑((AlgebraicGeometry.Proj.toLocallyRingedSpace 𝒜).restrict ⋯).toPresheafe …
    ⊢ ∀ (i : Nat) (x_1 : A), Membership.mem (𝒜 i) x_1 → Iff (Membership.mem (↑(Alg …
  -/
  intros i z hzi
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    m : Nat
    f_deg : Membership.mem (𝒜 m) f
    hm : LT.lt 0 m
    x : ↑↑((AlgebraicGeometry.Proj.toLocallyRingedSpace 𝒜).restrict ⋯).toPresheafe …
    i : Nat
    z : A
    hzi : Membership.mem (𝒜 i) z
    ⊢ Iff (Membership.mem (↑(AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.to …
  -/
  refine (FromSpec.mem_carrier_iff_of_mem f_deg hm _ _ hzi).trans ?_
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    m : Nat
    f_deg : Membership.mem (𝒜 m) f
    hm : LT.lt 0 m
    x : ↑↑((AlgebraicGeometry.Proj.toLocallyRingedSpace 𝒜).restrict ⋯).toPresheafe …
    i : Nat
    z : A
    hzi : Membership.mem (𝒜 i) z
    ⊢ Iff (Membership.mem ((AlgebraicGeometry.ProjIsoSpecTopComponent.toSpec 𝒜 f)  …
  -/
  exact (ToSpec.mk_mem_carrier _ _).trans (x.1.2.pow_mem_iff_mem m hm)
  /-
    🎉 no goals
  -/


lemma toSpec_injective {f : A} {m : ℕ} (f_deg : f ∈ 𝒜 m) (hm : 0 < m) :
    Function.Injective (toSpec 𝒜 f) := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    m : Nat
    f_deg : Membership.mem (𝒜 m) f
    hm : LT.lt 0 m
    ⊢ Function.Injective ⇑(AlgebraicGeometry.ProjIsoSpecTopComponent.toSpec 𝒜 f)
  -/
  intro x₁ x₂ h
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    m : Nat
    f_deg : Membership.mem (𝒜 m) f
    hm : LT.lt 0 m
    x₁ x₂ : ↑↑((AlgebraicGeometry.Proj.toLocallyRingedSpace 𝒜).restrict ⋯).toPresh …
    h : Eq ((AlgebraicGeometry.ProjIsoSpecTopComponent.toSpec 𝒜 f) x₁) ((Algebraic …
    ⊢ Eq x₁ x₂
  -/
  have := congr_arg (FromSpec.toFun f_deg hm) h
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    m : Nat
    f_deg : Membership.mem (𝒜 m) f
    hm : LT.lt 0 m
    x₁ x₂ : ↑↑((AlgebraicGeometry.Proj.toLocallyRingedSpace 𝒜).restrict ⋯).toPresh …
    h : Eq ((AlgebraicGeometry.ProjIsoSpecTopComponent.toSpec 𝒜 f) x₁) ((Algebraic …
    this : Eq (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.toFun f_deg hm ( …
    ⊢ Eq x₁ x₂
  -/
  rwa [fromSpec_toSpec, fromSpec_toSpec] at this
  /-
    🎉 no goals
  -/


lemma toSpec_surjective {f : A} {m : ℕ} (f_deg : f ∈ 𝒜 m) (hm : 0 < m) :
    Function.Surjective (toSpec 𝒜 f) :=
  Function.surjective_iff_hasRightInverse |>.mpr
    ⟨FromSpec.toFun f_deg hm, toSpec_fromSpec 𝒜 f_deg hm⟩


lemma toSpec_bijective {f : A} {m : ℕ} (f_deg : f ∈ 𝒜 m) (hm : 0 < m) :
    Function.Bijective (toSpec (𝒜 := 𝒜) (f := f)) :=
  ⟨toSpec_injective 𝒜 f_deg hm, toSpec_surjective 𝒜 f_deg hm⟩


variable {𝒜} in
lemma image_basicOpen_eq_basicOpen (a : A) (i : ℕ) :
    toSpec 𝒜 f '' (Subtype.val ⁻¹' (pbo (decompose 𝒜 a i) : Set (ProjectiveSpectrum 𝒜))) =
    (PrimeSpectrum.basicOpen (R := A⁰_ f) <|
      HomogeneousLocalization.mk
        ⟨m * i, ⟨decompose 𝒜 a i ^ m,
          (smul_eq_mul ℕ) ▸ SetLike.pow_mem_graded _ (Submodule.coe_mem _)⟩,
                   /-
                     R : Type u_1
                     A : Type u_2
                     inst✝³ : CommRing R
                     inst✝² : CommRing A
                     inst✝¹ : Algebra R A
                     𝒜 : Nat → Submodule R A
                     inst✝ : GradedAlgebra 𝒜
                     f : A
                     m : Nat
                     f_deg : Membership.mem (𝒜 m) f
                     hm : LT.lt 0 m
                     a : A
                     i : Nat
                     ⊢ Membership.mem (𝒜 (HMul.hMul m i)) (HPow.hPow f i)
                   -/
          ⟨f^i, by rw [mul_comm]; exact SetLike.pow_mem_graded _ f_deg⟩, ⟨i, rfl⟩⟩).1 :=
                                  /-
                                    🎉 no goals
                                  -/
  Set.preimage_injective.mpr (toSpec_surjective 𝒜 f_deg hm) <|
    Set.preimage_image_eq _ (toSpec_injective 𝒜 f_deg hm) ▸ by
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    m : Nat
    f_deg : Membership.mem (𝒜 m) f
    hm : LT.lt 0 m
    a : A
    i : Nat
    ⊢ Eq (Set.preimage Subtype.val ↑(ProjectiveSpectrum.basicOpen 𝒜 ↑(((DirectSum. …
  -/
  rw [Opens.carrier_eq_coe, toSpec_preimage_basicOpen, ProjectiveSpectrum.basicOpen_pow 𝒜 _ m hm]
  /-
    🎉 no goals
  -/


variable {𝒜} in
/-- The continuous function `Spec A⁰_f → Proj|D(f)` sending `q` to `{a | aᵢᵐ/fⁱ ∈ q}` where
`m` is the degree of `f` -/
def fromSpec {f : A} {m : ℕ} (f_deg : f ∈ 𝒜 m) (hm : 0 < m) :
    (Spec.T (A⁰_ f)) ⟶ (Proj.T| (pbo f)) where
  toFun := FromSpec.toFun f_deg hm
  continuous_toFun := by
    rw [isTopologicalBasis_subtype (ProjectiveSpectrum.isTopologicalBasis_basic_opens 𝒜) (pbo f).1
      |>.continuous_iff]
    /-
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      f : A
      m : Nat
      f_deg : Membership.mem (𝒜 m) f
      hm : LT.lt 0 m
      ⊢ ∀ (s : Set (Subtype (ProjectiveSpectrum.basicOpen 𝒜 f).carrier)), Membership …
    -/
    rintro s ⟨_, ⟨a, rfl⟩, rfl⟩
    have h₁ : Subtype.val (p := (pbo f).1) ⁻¹' (pbo a) =
        ⋃ i : ℕ, Subtype.val (p := (pbo f).1) ⁻¹' (pbo (decompose 𝒜 a i)) := by
      simp [ProjectiveSpectrum.basicOpen_eq_union_of_projection 𝒜 a]
    let e : _ ≃ _ :=
      ⟨FromSpec.toFun f_deg hm, ToSpec.toFun f, toSpec_fromSpec _ _ _, fromSpec_toSpec _ _ _⟩
    /-
      case intro.intro.intro
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      f : A
      m : Nat
      f_deg : Membership.mem (𝒜 m) f
      hm : LT.lt 0 m
      a : A
      h₁ : Eq (Set.preimage Subtype.val ↑(ProjectiveSpectrum.basicOpen 𝒜 a)) (Set.iU …
      e : Equiv ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Hom …
      ⊢ IsOpen (Set.preimage (AlgebraicGeometry.ProjIsoSpecTopComponent.FromSpec.toF …
    -/
    change IsOpen <| e ⁻¹' _
    /-
      case intro.intro.intro
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      f : A
      m : Nat
      f_deg : Membership.mem (𝒜 m) f
      hm : LT.lt 0 m
      a : A
      h₁ : Eq (Set.preimage Subtype.val ↑(ProjectiveSpectrum.basicOpen 𝒜 a)) (Set.iU …
      e : Equiv ↑↑(AlgebraicGeometry.Spec.locallyRingedSpaceObj (CommRingCat.of (Hom …
      ⊢ IsOpen (Set.preimage (⇑e) (Set.preimage Subtype.val ((fun r => ↑(ProjectiveS …
    -/
    rw [Set.preimage_equiv_eq_image_symm, h₁, Set.image_iUnion]
    exact isOpen_iUnion fun i ↦ toSpec.image_basicOpen_eq_basicOpen f_deg hm a i ▸
      PrimeSpectrum.isOpen_basicOpen


variable {𝒜} in
/--
The homeomorphism `Proj|D(f) ≅ Spec A⁰_f` defined by
- `φ : Proj|D(f) ⟶ Spec A⁰_f` by sending `x` to `A⁰_f ∩ span {g / 1 | g ∈ x}`
- `ψ : Spec A⁰_f ⟶ Proj|D(f)` by sending `q` to `{a | aᵢᵐ/fⁱ ∈ q}`.
-/
def projIsoSpecTopComponent {f : A} {m : ℕ} (f_deg : f ∈ 𝒜 m) (hm : 0 < m) :
    (Proj.T| (pbo f)) ≅ (Spec.T (A⁰_ f))  where
  hom := ProjIsoSpecTopComponent.toSpec 𝒜 f
  inv := ProjIsoSpecTopComponent.fromSpec f_deg hm
  hom_inv_id := ConcreteCategory.hom_ext _ _
    (ProjIsoSpecTopComponent.fromSpec_toSpec 𝒜 f_deg hm)
  inv_hom_id := ConcreteCategory.hom_ext _ _
    (ProjIsoSpecTopComponent.toSpec_fromSpec 𝒜 f_deg hm)


/--
The ring map from `A⁰_ f` to the local sections of the structure sheaf of the projective spectrum of
`A` on the basic open set `D(f)` defined by sending `s ∈ A⁰_f` to the section `x ↦ s` on `D(f)`.
-/
def awayToSection (f) : CommRingCat.of (A⁰_ f) ⟶ (structureSheaf 𝒜).1.obj (op (pbo f)) :=
  CommRingCat.ofHom
    -- Have to hint `S`, otherwise it gets unfolded to `structureSheafInType`
    -- causing `ext` to fail
    (S := (structureSheaf 𝒜).1.obj (op (pbo f)))
  { toFun s :=
      ⟨fun x ↦ HomogeneousLocalization.mapId 𝒜 (Submonoid.powers_le.mpr x.2) s, fun x ↦ by
        /-
          R : Type u_1
          A : Type u_2
          inst✝³ : CommRing R
          inst✝² : CommRing A
          inst✝¹ : Algebra R A
          𝒜 : Nat → Submodule R A
          inst✝ : GradedAlgebra 𝒜
          f : A
          s : HomogeneousLocalization.Away 𝒜 f
          x : Subtype fun x => Membership.mem (Opposite.unop { unop := ProjectiveSpectru …
          ⊢ Exists fun V => Exists fun x => Exists fun i => (AlgebraicGeometry.Projectiv …
        -/
        obtain ⟨s, rfl⟩ := HomogeneousLocalization.mk_surjective s
        /-
          case intro
          R : Type u_1
          A : Type u_2
          inst✝³ : CommRing R
          inst✝² : CommRing A
          inst✝¹ : Algebra R A
          𝒜 : Nat → Submodule R A
          inst✝ : GradedAlgebra 𝒜
          f : A
          x : Subtype fun x => Membership.mem (Opposite.unop { unop := ProjectiveSpectru …
          s : HomogeneousLocalization.NumDenSameDeg 𝒜 (Submonoid.powers f)
          ⊢ Exists fun V => Exists fun x => Exists fun i => (AlgebraicGeometry.Projectiv …
        -/
        obtain ⟨n, hn : f ^ n = s.den.1⟩ := s.den_mem
        exact ⟨_, x.2, 𝟙 _, s.1, s.2, s.3,
          fun x hsx ↦ x.2 (Ideal.IsPrime.mem_of_pow_mem inferInstance n (hn ▸ hsx)), fun _ ↦ rfl⟩⟩
                       /-
                         R : Type u_1
                         A : Type u_2
                         inst✝³ : CommRing R
                         inst✝² : CommRing A
                         inst✝¹ : Algebra R A
                         𝒜 : Nat → Submodule R A
                         inst✝ : GradedAlgebra 𝒜
                         f : A
                         x✝¹ x✝ : HomogeneousLocalization.Away 𝒜 f
                         ⊢ Eq ((↑{ toFun := fun s => ⟨fun x => (HomogeneousLocalization.mapId 𝒜 ⋯) s, ⋯ …
                       -/
                       /-
                         R : Type u_1
                         A : Type u_2
                         inst✝³ : CommRing R
                         inst✝² : CommRing A
                         inst✝¹ : Algebra R A
                         𝒜 : Nat → Submodule R A
                         inst✝ : GradedAlgebra 𝒜
                         f : A
                         x✝¹ x✝ : HomogeneousLocalization.Away 𝒜 f
                         ⊢ Eq ({ toFun := fun s => ⟨fun x => (HomogeneousLocalization.mapId 𝒜 ⋯) s, ⋯⟩, …
                       -/
    map_add' _ _ := by ext; simp only [map_add, HomogeneousLocalization.val_add, Proj.add_apply]
                   /-
                     R : Type u_1
                     A : Type u_2
                     inst✝³ : CommRing R
                     inst✝² : CommRing A
                     inst✝¹ : Algebra R A
                     𝒜 : Nat → Submodule R A
                     inst✝ : GradedAlgebra 𝒜
                     f : A
                     ⊢ Eq ((fun s => ⟨fun x => (HomogeneousLocalization.mapId 𝒜 ⋯) s, ⋯⟩) 1) 1
                   -/
                            /-
                              🎉 no goals
                            -/
                        /-
                          🎉 no goals
                        -/
                    /-
                      R : Type u_1
                      A : Type u_2
                      inst✝³ : CommRing R
                      inst✝² : CommRing A
                      inst✝¹ : Algebra R A
                      𝒜 : Nat → Submodule R A
                      inst✝ : GradedAlgebra 𝒜
                      f : A
                      ⊢ Eq ((↑{ toFun := fun s => ⟨fun x => (HomogeneousLocalization.mapId 𝒜 ⋯) s, ⋯ …
                    -/
                            /-
                              🎉 no goals
                            -/
                         /-
                           🎉 no goals
                         -/
    map_mul' _ _ := by ext; simp only [map_mul, HomogeneousLocalization.val_mul, Proj.mul_apply]
    map_zero' := by ext; simp only [map_zero, HomogeneousLocalization.val_zero, Proj.zero_apply]
    map_one' := by ext; simp only [map_one, HomogeneousLocalization.val_one, Proj.one_apply] }


lemma awayToSection_germ (f x hx) :
    awayToSection 𝒜 f ≫ (structureSheaf 𝒜).presheaf.germ _ x hx =
      CommRingCat.ofHom (HomogeneousLocalization.mapId 𝒜 (Submonoid.powers_le.mpr hx)) ≫
        (Proj.stalkIso' 𝒜 x).toCommRingCatIso.inv := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    x : ↑(ProjectiveSpectrum.top 𝒜)
    hx : Membership.mem (ProjectiveSpectrum.basicOpen 𝒜 f) x
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.ProjectiveSpectrum …
  -/
  ext z
  /-
    case hf.a
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    x : ↑(ProjectiveSpectrum.top 𝒜)
    hx : Membership.mem (ProjectiveSpectrum.basicOpen 𝒜 f) x
    z : ↑(CommRingCat.of (HomogeneousLocalization.Away 𝒜 f))
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.ProjectiveSpectru …
  -/
  apply (Proj.stalkIso' 𝒜 x).eq_symm_apply.mpr
  /-
    case hf.a
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    x : ↑(ProjectiveSpectrum.top 𝒜)
    hx : Membership.mem (ProjectiveSpectrum.basicOpen 𝒜 f) x
    z : ↑(CommRingCat.of (HomogeneousLocalization.Away 𝒜 f))
    ⊢ Eq ((AlgebraicGeometry.Proj.stalkIso' 𝒜 x).toEquiv ((CategoryTheory.Category …
  -/
  apply Proj.stalkIso'_germ
  /-
    🎉 no goals
  -/


lemma awayToSection_apply (f : A) (x p) :
    (((ProjectiveSpectrum.Proj.awayToSection 𝒜 f).1 x).val p).val =
      IsLocalization.map (M := Submonoid.powers f) (T := p.1.1.toIdeal.primeCompl) _
        (RingHom.id _) (Submonoid.powers_le.mpr p.2) x.val := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    x : ↑(CommRingCat.of (HomogeneousLocalization.Away 𝒜 f))
    p : Subtype fun x => Membership.mem (Opposite.unop { unop := ProjectiveSpectru …
    ⊢ Eq (HomogeneousLocalization.val (↑((AlgebraicGeometry.ProjectiveSpectrum.Pro …
  -/
  obtain ⟨x, rfl⟩ := HomogeneousLocalization.mk_surjective x
  /-
    case intro
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    p : Subtype fun x => Membership.mem (Opposite.unop { unop := ProjectiveSpectru …
    x : HomogeneousLocalization.NumDenSameDeg 𝒜 (Submonoid.powers f)
    ⊢ Eq (HomogeneousLocalization.val (↑((AlgebraicGeometry.ProjectiveSpectrum.Pro …
  -/
  show (HomogeneousLocalization.mapId 𝒜 _ _).val = _
  /-
    case intro
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    p : Subtype fun x => Membership.mem (Opposite.unop { unop := ProjectiveSpectru …
    x : HomogeneousLocalization.NumDenSameDeg 𝒜 (Submonoid.powers f)
    ⊢ Eq ((HomogeneousLocalization.mapId 𝒜 ⋯) (HomogeneousLocalization.mk x)).val  …
  -/
  dsimp [HomogeneousLocalization.mapId, HomogeneousLocalization.map]
  /-
    case intro
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    p : Subtype fun x => Membership.mem (Opposite.unop { unop := ProjectiveSpectru …
    x : HomogeneousLocalization.NumDenSameDeg 𝒜 (Submonoid.powers f)
    ⊢ Eq (Localization.mk ↑x.num ⟨↑x.den, ⋯⟩) ((IsLocalization.map (Localization ( …
  -/
  rw [Localization.mk_eq_mk', Localization.mk_eq_mk', IsLocalization.map_mk']
  /-
    case intro
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    p : Subtype fun x => Membership.mem (Opposite.unop { unop := ProjectiveSpectru …
    x : HomogeneousLocalization.NumDenSameDeg 𝒜 (Submonoid.powers f)
    ⊢ Eq (IsLocalization.mk' (Localization (↑p).asHomogeneousIdeal.toIdeal.primeCo …
  -/
  rfl
  /-
    🎉 no goals
  -/


/--
The ring map from `A⁰_ f` to the global sections of the structure sheaf of the projective spectrum
of `A` restricted to the basic open set `D(f)`.

Mathematically, the map is the same as `awayToSection`.
-/
def awayToΓ (f) : CommRingCat.of (A⁰_ f) ⟶ LocallyRingedSpace.Γ.obj (op <| Proj| pbo f) :=
  awayToSection 𝒜 f ≫ (ProjectiveSpectrum.Proj.structureSheaf 𝒜).1.map
    (homOfLE (Opens.isOpenEmbedding_obj_top _).le).op


lemma awayToΓ_ΓToStalk (f) (x) :
    awayToΓ 𝒜 f ≫ (Proj| pbo f).presheaf.Γgerm x =
      CommRingCat.ofHom (HomogeneousLocalization.mapId 𝒜 (Submonoid.powers_le.mpr x.2)) ≫
      (Proj.stalkIso' 𝒜 x.1).toCommRingCatIso.inv ≫
      ((Proj.toLocallyRingedSpace 𝒜).restrictStalkIso (Opens.isOpenEmbedding _) x).inv := by
  rw [awayToΓ, Category.assoc, ← Category.assoc _ (Iso.inv _),
    Iso.eq_comp_inv, Category.assoc, Category.assoc, Presheaf.Γgerm]
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    x : ↑↑((AlgebraicGeometry.Proj.toLocallyRingedSpace 𝒜).restrict ⋯).toPresheafe …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.ProjectiveSpectrum …
  -/
  rw [LocallyRingedSpace.restrictStalkIso_hom_eq_germ]
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    x : ↑↑((AlgebraicGeometry.Proj.toLocallyRingedSpace 𝒜).restrict ⋯).toPresheafe …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.ProjectiveSpectrum …
  -/
  simp only [Proj.toLocallyRingedSpace, Proj.toSheafedSpace]
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    x : ↑↑((AlgebraicGeometry.Proj.toLocallyRingedSpace 𝒜).restrict ⋯).toPresheafe …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.ProjectiveSpectrum …
  -/
  rw [Presheaf.germ_res, awayToSection_germ]
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    x : ↑↑((AlgebraicGeometry.Proj.toLocallyRingedSpace 𝒜).restrict ⋯).toPresheafe …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CommRingCat.ofHom (HomogeneousLocali …
  -/
  rfl
  /-
    🎉 no goals
  -/


/--
The morphism of locally ringed space from `Proj|D(f)` to `Spec A⁰_f` induced by the ring map
`A⁰_ f → Γ(Proj, D(f))` under the gamma spec adjunction.
-/
def toSpec (f) : (Proj| pbo f) ⟶ Spec (A⁰_ f) :=
  ΓSpec.locallyRingedSpaceAdjunction.homEquiv (Proj| pbo f) (op (CommRingCat.of <| A⁰_ f))
    (awayToΓ 𝒜 f).op


lemma toSpec_base_apply_eq_comap {f} (x : Proj| pbo f) :
    (toSpec 𝒜 f).base x = PrimeSpectrum.comap (mapId 𝒜 (Submonoid.powers_le.mpr x.2))
      (closedPoint (AtPrime 𝒜 x.1.asHomogeneousIdeal.toIdeal)) := by
  show PrimeSpectrum.comap (awayToΓ 𝒜 f ≫ (Proj| pbo f).presheaf.Γgerm x).hom
        (IsLocalRing.closedPoint ((Proj| pbo f).presheaf.stalk x)) = _
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    x : ↑((AlgebraicGeometry.Proj.toLocallyRingedSpace 𝒜).restrict ⋯).toTopCat
    ⊢ Eq ((PrimeSpectrum.comap (CategoryTheory.CategoryStruct.comp (AlgebraicGeome …
  -/
  rw [awayToΓ_ΓToStalk, CommRingCat.hom_comp, PrimeSpectrum.comap_comp]
  exact congr(PrimeSpectrum.comap _ $(@IsLocalRing.comap_closedPoint
    (HomogeneousLocalization.AtPrime 𝒜 x.1.asHomogeneousIdeal.toIdeal) _ _
    ((Proj| pbo f).presheaf.stalk x) _ _ _ (isLocalHom_of_isIso _)))


lemma toSpec_base_apply_eq {f} (x : Proj| pbo f) :
    (toSpec 𝒜 f).base x = ProjIsoSpecTopComponent.toSpec 𝒜 f x :=
  toSpec_base_apply_eq_comap 𝒜 x |>.trans <| PrimeSpectrum.ext <| Ideal.ext fun z =>
  show ¬ IsUnit _ ↔ z ∈ ProjIsoSpecTopComponent.ToSpec.carrier _ by
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    x : ↑((AlgebraicGeometry.Proj.toLocallyRingedSpace 𝒜).restrict ⋯).toTopCat
    z : HomogeneousLocalization 𝒜 (Submonoid.powers f)
    ⊢ Iff (Not (IsUnit ((HomogeneousLocalization.mapId 𝒜 ⋯) z))) (Membership.mem ( …
  -/
  obtain ⟨z, rfl⟩ := z.mk_surjective
  rw [← HomogeneousLocalization.isUnit_iff_isUnit_val,
    ProjIsoSpecTopComponent.ToSpec.mk_mem_carrier, HomogeneousLocalization.map_mk,
    HomogeneousLocalization.val_mk, Localization.mk_eq_mk',
    IsLocalization.AtPrime.isUnit_mk'_iff]
  /-
    case intro
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    x : ↑((AlgebraicGeometry.Proj.toLocallyRingedSpace 𝒜).restrict ⋯).toTopCat
    z : HomogeneousLocalization.NumDenSameDeg 𝒜 (Submonoid.powers f)
    ⊢ Iff (Not (Membership.mem (↑x).asHomogeneousIdeal.toIdeal.primeCompl ↑{ deg : …
  -/
  exact not_not
  /-
    🎉 no goals
  -/


lemma toSpec_base_isIso {f} {m} (f_deg : f ∈ 𝒜 m) (hm : 0 < m) :
    IsIso (toSpec 𝒜 f).base := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    m : Nat
    f_deg : Membership.mem (𝒜 m) f
    hm : LT.lt 0 m
    ⊢ CategoryTheory.IsIso (AlgebraicGeometry.ProjectiveSpectrum.Proj.toSpec 𝒜 f). …
  -/
  convert (projIsoSpecTopComponent f_deg hm).isIso_hom
  /-
    case h.e'_5
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    m : Nat
    f_deg : Membership.mem (𝒜 m) f
    hm : LT.lt 0 m
    ⊢ Eq (AlgebraicGeometry.ProjectiveSpectrum.Proj.toSpec 𝒜 f).base (AlgebraicGeo …
  -/
  exact DFunLike.ext _ _ <| toSpec_base_apply_eq 𝒜
  /-
    🎉 no goals
  -/


lemma mk_mem_toSpec_base_apply {f} (x : Proj| pbo f)
    (z : NumDenSameDeg 𝒜 (.powers f)) :
    HomogeneousLocalization.mk z ∈ ((toSpec 𝒜 f).base x).asIdeal ↔
      z.num.1 ∈ x.1.asHomogeneousIdeal :=
  (toSpec_base_apply_eq 𝒜 x).symm ▸ ProjIsoSpecTopComponent.ToSpec.mk_mem_carrier _ _


lemma toSpec_preimage_basicOpen {f}
    (t : NumDenSameDeg 𝒜 (.powers f)) :
    (Opens.map (toSpec 𝒜 f).base).obj (sbo (HomogeneousLocalization.mk t)) =
      Opens.comap ⟨_, continuous_subtype_val⟩ (pbo t.num.1) :=
  Opens.ext <| Opens.map_coe _ _ ▸ by
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    t : HomogeneousLocalization.NumDenSameDeg 𝒜 (Submonoid.powers f)
    ⊢ Eq (Set.preimage ⇑(AlgebraicGeometry.ProjectiveSpectrum.Proj.toSpec 𝒜 f).bas …
  -/
  convert (ProjIsoSpecTopComponent.ToSpec.preimage_basicOpen f t)
  /-
    case h.e'_2.h.e'_3
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    t : HomogeneousLocalization.NumDenSameDeg 𝒜 (Submonoid.powers f)
    ⊢ Eq (⇑(AlgebraicGeometry.ProjectiveSpectrum.Proj.toSpec 𝒜 f).base) (Algebraic …
  -/
  exact funext fun _ => toSpec_base_apply_eq _ _
  /-
    🎉 no goals
  -/


@[reassoc]
lemma toOpen_toSpec_val_c_app (f) (U) :
    StructureSheaf.toOpen (A⁰_ f) U.unop ≫ (toSpec 𝒜 f).c.app U =
      awayToΓ 𝒜 f ≫ (Proj| pbo f).presheaf.map (homOfLE le_top).op :=
               /-
                 R : Type u_1
                 A : Type u_2
                 inst✝³ : CommRing R
                 inst✝² : CommRing A
                 inst✝¹ : Algebra R A
                 𝒜 : Nat → Submodule R A
                 inst✝ : GradedAlgebra 𝒜
                 f : A
                 U : Opposite (TopologicalSpace.Opens ↑↑(AlgebraicGeometry.Spec.locallyRingedSp …
                 ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.StructureSheaf.toO …
               -/
  Eq.trans (by congr) <| ΓSpec.toOpen_comp_locallyRingedSpaceAdjunction_homEquiv_app _ U
               /-
                 🎉 no goals
               -/


@[reassoc]
lemma toStalk_stalkMap_toSpec (f) (x) :
    StructureSheaf.toStalk _ _ ≫ (toSpec 𝒜 f).stalkMap x =
      awayToΓ 𝒜 f ≫ (Proj| pbo f).presheaf.Γgerm x := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    x : ↑((AlgebraicGeometry.Proj.toLocallyRingedSpace 𝒜).restrict ⋯).toTopCat
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.StructureSheaf.toS …
  -/
  rw [StructureSheaf.toStalk, Category.assoc]
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    x : ↑((AlgebraicGeometry.Proj.toLocallyRingedSpace 𝒜).restrict ⋯).toTopCat
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.StructureSheaf.toO …
  -/
  simp_rw [← Spec.locallyRingedSpaceObj_presheaf']
  rw [LocallyRingedSpace.stalkMap_germ (toSpec 𝒜 f),
    toOpen_toSpec_val_c_app_assoc, Presheaf.germ_res]
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    x : ↑((AlgebraicGeometry.Proj.toLocallyRingedSpace 𝒜).restrict ⋯).toTopCat
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.ProjectiveSpectrum …
  -/
  rfl
  /-
    🎉 no goals
  -/


/--
If `x` is a point in the basic open set `D(f)` where `f` is a homogeneous element of positive
degree, then the homogeneously localized ring `A⁰ₓ` has the universal property of the localization
of `A⁰_f` at `φ(x)` where `φ : Proj|D(f) ⟶ Spec A⁰_f` is the morphism of locally ringed space
constructed as above.
-/
lemma isLocalization_atPrime (f) (x : pbo f) {m} (f_deg : f ∈ 𝒜 m) (hm : 0 < m) :
    @IsLocalization (Away 𝒜 f) _ ((toSpec 𝒜 f).base x).asIdeal.primeCompl
      (AtPrime 𝒜 x.1.asHomogeneousIdeal.toIdeal) _
      (mapId 𝒜 (Submonoid.powers_le.mpr x.2)).toAlgebra := by
  letI : Algebra (Away 𝒜 f) (AtPrime 𝒜 x.1.asHomogeneousIdeal.toIdeal) :=
    (mapId 𝒜 (Submonoid.powers_le.mpr x.2)).toAlgebra
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    x : Subtype fun x => Membership.mem (ProjectiveSpectrum.basicOpen 𝒜 f) x
    m : Nat
    f_deg : Membership.mem (𝒜 m) f
    hm : LT.lt 0 m
    this : Algebra (HomogeneousLocalization.Away 𝒜 f) (HomogeneousLocalization.AtP …
    ⊢ IsLocalization ((AlgebraicGeometry.ProjectiveSpectrum.Proj.toSpec 𝒜 f).base  …
  -/
  constructor
    /-
      case map_units'
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      f : A
      x : Subtype fun x => Membership.mem (ProjectiveSpectrum.basicOpen 𝒜 f) x
      m : Nat
      f_deg : Membership.mem (𝒜 m) f
      hm : LT.lt 0 m
      this : Algebra (HomogeneousLocalization.Away 𝒜 f) (HomogeneousLocalization.AtP …
      ⊢ ∀ (y : Subtype fun x_1 => Membership.mem ((AlgebraicGeometry.ProjectiveSpect …
    -/
  · rintro ⟨y, hy⟩
    /-
      case map_units'.mk
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      f : A
      x : Subtype fun x => Membership.mem (ProjectiveSpectrum.basicOpen 𝒜 f) x
      m : Nat
      f_deg : Membership.mem (𝒜 m) f
      hm : LT.lt 0 m
      this : Algebra (HomogeneousLocalization.Away 𝒜 f) (HomogeneousLocalization.AtP …
      y : HomogeneousLocalization.Away 𝒜 f
      hy : Membership.mem ((AlgebraicGeometry.ProjectiveSpectrum.Proj.toSpec 𝒜 f).ba …
      ⊢ IsUnit ((algebraMap (HomogeneousLocalization.Away 𝒜 f) (HomogeneousLocalizat …
    -/
    obtain ⟨y, rfl⟩ := HomogeneousLocalization.mk_surjective y
    refine isUnit_of_mul_eq_one _
      (.mk ⟨y.deg, y.den, y.num, (mk_mem_toSpec_base_apply _ _ _).not.mp hy⟩) <| val_injective _ ?_
    simp only [RingHom.algebraMap_toAlgebra, map_mk, RingHom.id_apply, val_mul, val_mk, mk_eq_mk',
      val_one, IsLocalization.mk'_mul_mk'_eq_one']
    /-
      case surj'
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      f : A
      x : Subtype fun x => Membership.mem (ProjectiveSpectrum.basicOpen 𝒜 f) x
      m : Nat
      f_deg : Membership.mem (𝒜 m) f
      hm : LT.lt 0 m
      this : Algebra (HomogeneousLocalization.Away 𝒜 f) (HomogeneousLocalization.AtP …
      ⊢ ∀ (z : HomogeneousLocalization.AtPrime 𝒜 (↑x).asHomogeneousIdeal.toIdeal), E …
    -/
  · intro z
    /-
      case surj'
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      f : A
      x : Subtype fun x => Membership.mem (ProjectiveSpectrum.basicOpen 𝒜 f) x
      m : Nat
      f_deg : Membership.mem (𝒜 m) f
      hm : LT.lt 0 m
      this : Algebra (HomogeneousLocalization.Away 𝒜 f) (HomogeneousLocalization.AtP …
      z : HomogeneousLocalization.AtPrime 𝒜 (↑x).asHomogeneousIdeal.toIdeal
      ⊢ Exists fun x_1 => Eq (HMul.hMul z ((algebraMap (HomogeneousLocalization.Away …
    -/
    obtain ⟨⟨i, a, ⟨b, hb⟩, (hb' : b ∉ x.1.1)⟩, rfl⟩ := z.mk_surjective
    refine ⟨⟨HomogeneousLocalization.mk ⟨i * m, ⟨a * b ^ (m - 1), ?_⟩,
        ⟨f ^ i, SetLike.pow_mem_graded _ f_deg⟩, ⟨_, rfl⟩⟩,
      ⟨HomogeneousLocalization.mk ⟨i * m, ⟨b ^ m, mul_comm m i ▸ SetLike.pow_mem_graded _ hb⟩,
        ⟨f ^ i, SetLike.pow_mem_graded _ f_deg⟩, ⟨_, rfl⟩⟩,
        (mk_mem_toSpec_base_apply _ _ _).not.mpr <| x.1.1.toIdeal.primeCompl.pow_mem hb' m⟩⟩,
        val_injective _ ?_⟩
      /-
        case surj'.intro.mk.mk.refine_1
        R : Type u_1
        A : Type u_2
        inst✝³ : CommRing R
        inst✝² : CommRing A
        inst✝¹ : Algebra R A
        𝒜 : Nat → Submodule R A
        inst✝ : GradedAlgebra 𝒜
        f : A
        x : Subtype fun x => Membership.mem (ProjectiveSpectrum.basicOpen 𝒜 f) x
        m : Nat
        f_deg : Membership.mem (𝒜 m) f
        hm : LT.lt 0 m
        this : Algebra (HomogeneousLocalization.Away 𝒜 f) (HomogeneousLocalization.AtP …
        i : Nat
        a : Subtype fun x => Membership.mem (𝒜 i) x
        b : A
        hb : Membership.mem (𝒜 i) b
        hb' : Not (Membership.mem (↑x).asHomogeneousIdeal b)
        ⊢ Membership.mem (𝒜 (HMul.hMul i m)) (HMul.hMul (↑a) (HPow.hPow b (HSub.hSub m …
      -/
    · convert SetLike.mul_mem_graded a.2 (SetLike.pow_mem_graded (m - 1) hb) using 2
      /-
        case h.e'_4.h.e'_1
        R : Type u_1
        A : Type u_2
        inst✝³ : CommRing R
        inst✝² : CommRing A
        inst✝¹ : Algebra R A
        𝒜 : Nat → Submodule R A
        inst✝ : GradedAlgebra 𝒜
        f : A
        x : Subtype fun x => Membership.mem (ProjectiveSpectrum.basicOpen 𝒜 f) x
        m : Nat
        f_deg : Membership.mem (𝒜 m) f
        hm : LT.lt 0 m
        this : Algebra (HomogeneousLocalization.Away 𝒜 f) (HomogeneousLocalization.AtP …
        i : Nat
        a : Subtype fun x => Membership.mem (𝒜 i) x
        b : A
        hb : Membership.mem (𝒜 i) b
        hb' : Not (Membership.mem (↑x).asHomogeneousIdeal b)
        ⊢ Eq (HMul.hMul i m) (HAdd.hAdd i (HSMul.hSMul (HSub.hSub m 1) i))
      -/
      rw [← succ_nsmul', tsub_add_cancel_of_le (by omega), mul_comm, smul_eq_mul]
      /-
        🎉 no goals
      -/
    · simp only [RingHom.algebraMap_toAlgebra, map_mk, RingHom.id_apply, val_mul, val_mk,
        mk_eq_mk', ← IsLocalization.mk'_mul, Submonoid.mk_mul_mk, IsLocalization.mk'_eq_iff_eq]
      /-
        case surj'.intro.mk.mk.refine_2
        R : Type u_1
        A : Type u_2
        inst✝³ : CommRing R
        inst✝² : CommRing A
        inst✝¹ : Algebra R A
        𝒜 : Nat → Submodule R A
        inst✝ : GradedAlgebra 𝒜
        f : A
        x : Subtype fun x => Membership.mem (ProjectiveSpectrum.basicOpen 𝒜 f) x
        m : Nat
        f_deg : Membership.mem (𝒜 m) f
        hm : LT.lt 0 m
        this : Algebra (HomogeneousLocalization.Away 𝒜 f) (HomogeneousLocalization.AtP …
        i : Nat
        a : Subtype fun x => Membership.mem (𝒜 i) x
        b : A
        hb : Membership.mem (𝒜 i) b
        hb' : Not (Membership.mem (↑x).asHomogeneousIdeal b)
        ⊢ Eq ((algebraMap A (Localization (↑x).asHomogeneousIdeal.toIdeal.primeCompl)) …
      -/
      rw [mul_comm b, mul_mul_mul_comm, ← pow_succ', mul_assoc, tsub_add_cancel_of_le (by omega)]
      /-
        🎉 no goals
      -/
    /-
      case exists_of_eq
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      f : A
      x : Subtype fun x => Membership.mem (ProjectiveSpectrum.basicOpen 𝒜 f) x
      m : Nat
      f_deg : Membership.mem (𝒜 m) f
      hm : LT.lt 0 m
      this : Algebra (HomogeneousLocalization.Away 𝒜 f) (HomogeneousLocalization.AtP …
      ⊢ ∀ {x_1 y : HomogeneousLocalization.Away 𝒜 f}, Eq ((algebraMap (HomogeneousLo …
    -/
  · intros y z e
    /-
      case exists_of_eq
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      f : A
      x : Subtype fun x => Membership.mem (ProjectiveSpectrum.basicOpen 𝒜 f) x
      m : Nat
      f_deg : Membership.mem (𝒜 m) f
      hm : LT.lt 0 m
      this : Algebra (HomogeneousLocalization.Away 𝒜 f) (HomogeneousLocalization.AtP …
      y z : HomogeneousLocalization.Away 𝒜 f
      e : Eq ((algebraMap (HomogeneousLocalization.Away 𝒜 f) (HomogeneousLocalizatio …
      ⊢ Exists fun c => Eq (HMul.hMul (↑c) y) (HMul.hMul (↑c) z)
    -/
    obtain ⟨y, rfl⟩ := HomogeneousLocalization.mk_surjective y
    /-
      case exists_of_eq.intro
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      f : A
      x : Subtype fun x => Membership.mem (ProjectiveSpectrum.basicOpen 𝒜 f) x
      m : Nat
      f_deg : Membership.mem (𝒜 m) f
      hm : LT.lt 0 m
      this : Algebra (HomogeneousLocalization.Away 𝒜 f) (HomogeneousLocalization.AtP …
      z : HomogeneousLocalization.Away 𝒜 f
      y : HomogeneousLocalization.NumDenSameDeg 𝒜 (Submonoid.powers f)
      e : Eq ((algebraMap (HomogeneousLocalization.Away 𝒜 f) (HomogeneousLocalizatio …
      ⊢ Exists fun c => Eq (HMul.hMul (↑c) (HomogeneousLocalization.mk y)) (HMul.hMu …
    -/
    obtain ⟨z, rfl⟩ := HomogeneousLocalization.mk_surjective z
    obtain ⟨i, c, hc, hc', e⟩ : ∃ i, ∃ c ∈ 𝒜 i, c ∉ x.1.asHomogeneousIdeal ∧
        c * (z.den.1 * y.num.1) = c * (y.den.1 * z.num.1) := by
      apply_fun HomogeneousLocalization.val at e
      simp only [RingHom.algebraMap_toAlgebra, map_mk, RingHom.id_apply, val_mk, mk_eq_mk',
        IsLocalization.mk'_eq_iff_eq] at e
      obtain ⟨⟨c, hcx⟩, hc⟩ := IsLocalization.exists_of_eq (M := x.1.1.toIdeal.primeCompl) e
      obtain ⟨i, hi⟩ := not_forall.mp ((x.1.1.isHomogeneous.mem_iff _).not.mp hcx)
      refine ⟨i, _, (decompose 𝒜 c i).2, hi, ?_⟩
      apply_fun fun x ↦ (decompose 𝒜 x (i + z.deg + y.deg)).1 at hc
      conv_rhs at hc => rw [add_right_comm]
      rwa [← mul_assoc, coe_decompose_mul_add_of_right_mem, coe_decompose_mul_add_of_right_mem,
        ← mul_assoc, coe_decompose_mul_add_of_right_mem, coe_decompose_mul_add_of_right_mem,
        mul_assoc, mul_assoc] at hc
      exacts [y.den.2, z.num.2, z.den.2, y.num.2]

    refine ⟨⟨HomogeneousLocalization.mk ⟨m * i, ⟨c ^ m, SetLike.pow_mem_graded _ hc⟩,
      ⟨f ^ i, mul_comm m i ▸ SetLike.pow_mem_graded _ f_deg⟩, ⟨_, rfl⟩⟩,
      (mk_mem_toSpec_base_apply _ _ _).not.mpr <| x.1.1.toIdeal.primeCompl.pow_mem hc' _⟩,
      val_injective _ ?_⟩
    simp only [val_mul, val_mk, mk_eq_mk', ← IsLocalization.mk'_mul, Submonoid.mk_mul_mk,
      IsLocalization.mk'_eq_iff_eq, mul_assoc]
    /-
      case exists_of_eq.intro.intro.intro.intro.intro.intro
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      f : A
      x : Subtype fun x => Membership.mem (ProjectiveSpectrum.basicOpen 𝒜 f) x
      m : Nat
      f_deg : Membership.mem (𝒜 m) f
      hm : LT.lt 0 m
      this : Algebra (HomogeneousLocalization.Away 𝒜 f) (HomogeneousLocalization.AtP …
      y z : HomogeneousLocalization.NumDenSameDeg 𝒜 (Submonoid.powers f)
      e✝ : Eq ((algebraMap (HomogeneousLocalization.Away 𝒜 f) (HomogeneousLocalizati …
      i : Nat
      c : A
      hc : Membership.mem (𝒜 i) c
      hc' : Not (Membership.mem (↑x).asHomogeneousIdeal c)
      e : Eq (HMul.hMul c (HMul.hMul ↑z.den ↑y.num)) (HMul.hMul c (HMul.hMul ↑y.den  …
      ⊢ Eq ((algebraMap A (Localization (Submonoid.powers f))) (HMul.hMul (HPow.hPow …
    -/
    congr 2
    rw [mul_left_comm, mul_left_comm y.den.1, ← tsub_add_cancel_of_le (show 1 ≤ m from hm),
      pow_succ, mul_assoc, mul_assoc, e]


/--
For an element `f ∈ A` with positive degree and a homogeneous ideal in `D(f)`, we have that the
stalk of `Spec A⁰_ f` at `y` is isomorphic to `A⁰ₓ` where `y` is the point in `Proj` corresponding
to `x`.
-/
def specStalkEquiv (f) (x : pbo f) {m} (f_deg : f ∈ 𝒜 m) (hm : 0 < m) :
    (Spec.structureSheaf (A⁰_ f)).presheaf.stalk ((toSpec 𝒜 f).base x) ≅
      CommRingCat.of (AtPrime 𝒜 x.1.asHomogeneousIdeal.toIdeal) :=
  letI : Algebra (Away 𝒜 f) (AtPrime 𝒜 x.1.asHomogeneousIdeal.toIdeal) :=
    (mapId 𝒜 (Submonoid.powers_le.mpr x.2)).toAlgebra
  haveI := isLocalization_atPrime 𝒜 f x f_deg hm
  (IsLocalization.algEquiv
    (R := A⁰_ f)
    (M := ((toSpec 𝒜 f).base x).asIdeal.primeCompl)
    (S := (Spec.structureSheaf (A⁰_ f)).presheaf.stalk ((toSpec 𝒜 f).base x))
    (Q := AtPrime 𝒜 x.1.asHomogeneousIdeal.toIdeal)).toRingEquiv.toCommRingCatIso


lemma toStalk_specStalkEquiv (f) (x : pbo f) {m} (f_deg : f ∈ 𝒜 m) (hm : 0 < m) :
    StructureSheaf.toStalk (A⁰_ f) ((toSpec 𝒜 f).base x) ≫ (specStalkEquiv 𝒜 f x f_deg hm).hom =
      CommRingCat.ofHom (mapId _ <| Submonoid.powers_le.mpr x.2) :=
  letI : Algebra (Away 𝒜 f) (AtPrime 𝒜 x.1.asHomogeneousIdeal.toIdeal) :=
    (mapId 𝒜 (Submonoid.powers_le.mpr x.2)).toAlgebra
  letI := isLocalization_atPrime 𝒜 f x f_deg hm
  CommRingCat.hom_ext (IsLocalization.algEquiv
    (R := A⁰_ f)
    (M := ((toSpec 𝒜 f).base x).asIdeal.primeCompl)
    (S := (Spec.structureSheaf (A⁰_ f)).presheaf.stalk ((toSpec 𝒜 f).base x))
    (Q := AtPrime 𝒜 x.1.asHomogeneousIdeal.toIdeal)).toAlgHom.comp_algebraMap


lemma stalkMap_toSpec (f) (x : pbo f) {m} (f_deg : f ∈ 𝒜 m) (hm : 0 < m) :
    (toSpec 𝒜 f).stalkMap x =
      (specStalkEquiv 𝒜 f x f_deg hm).hom ≫ (Proj.stalkIso' 𝒜 x.1).toCommRingCatIso.inv ≫
      ((Proj.toLocallyRingedSpace 𝒜).restrictStalkIso (Opens.isOpenEmbedding _) x).inv :=
  CommRingCat.hom_ext <|
    IsLocalization.ringHom_ext (R := A⁰_ f) ((toSpec 𝒜 f).base x).asIdeal.primeCompl
      (S := (Spec.structureSheaf (A⁰_ f)).presheaf.stalk ((toSpec 𝒜 f).base x)) <|
      CommRingCat.hom_ext_iff.mp <|
        (toStalk_stalkMap_toSpec _ _ _).trans <| by
        /-
          R : Type u_1
          A : Type u_2
          inst✝³ : CommRing R
          inst✝² : CommRing A
          inst✝¹ : Algebra R A
          𝒜 : Nat → Submodule R A
          inst✝ : GradedAlgebra 𝒜
          f : A
          x : Subtype fun x => Membership.mem (ProjectiveSpectrum.basicOpen 𝒜 f) x
          m : Nat
          f_deg : Membership.mem (𝒜 m) f
          hm : LT.lt 0 m
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.ProjectiveSpectrum …
        -/
        rw [awayToΓ_ΓToStalk, ← toStalk_specStalkEquiv, Category.assoc]; rfl
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


lemma isIso_toSpec (f) {m} (f_deg : f ∈ 𝒜 m) (hm : 0 < m) :
    IsIso (toSpec 𝒜 f) := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    m : Nat
    f_deg : Membership.mem (𝒜 m) f
    hm : LT.lt 0 m
    ⊢ CategoryTheory.IsIso (AlgebraicGeometry.ProjectiveSpectrum.Proj.toSpec 𝒜 f)
  -/
  haveI : IsIso (toSpec 𝒜 f).base := toSpec_base_isIso 𝒜 f_deg hm
  haveI (x) : IsIso ((toSpec 𝒜 f).stalkMap x) := by
    rw [stalkMap_toSpec 𝒜 f x f_deg hm]; infer_instance
  haveI : LocallyRingedSpace.IsOpenImmersion (toSpec 𝒜 f) :=
    LocallyRingedSpace.IsOpenImmersion.of_stalk_iso (toSpec 𝒜 f)
      (TopCat.homeoOfIso (asIso <| (toSpec 𝒜 f).base)).isOpenEmbedding
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    f : A
    m : Nat
    f_deg : Membership.mem (𝒜 m) f
    hm : LT.lt 0 m
    this✝¹ : CategoryTheory.IsIso (AlgebraicGeometry.ProjectiveSpectrum.Proj.toSpe …
    this✝ : ∀ (x : ↑((AlgebraicGeometry.Proj.toLocallyRingedSpace 𝒜).restrict ⋯).t …
    this : AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion (AlgebraicGeometry …
    ⊢ CategoryTheory.IsIso (AlgebraicGeometry.ProjectiveSpectrum.Proj.toSpec 𝒜 f)
  -/
  exact LocallyRingedSpace.IsOpenImmersion.to_iso _
  /-
    🎉 no goals
  -/


open ProjectiveSpectrum.Proj in
/--
If `f ∈ A` is a homogeneous element of positive degree, then the projective spectrum restricted to
`D(f)` as a locally ringed space is isomorphic to `Spec A⁰_f`.
-/
def projIsoSpec (f) {m} (f_deg : f ∈ 𝒜 m) (hm : 0 < m) :
    (Proj| pbo f) ≅ (Spec (A⁰_ f)) :=
  @asIso _ _ _ _ (f := toSpec 𝒜 f) (isIso_toSpec 𝒜 f f_deg hm)


/--
This is the scheme `Proj(A)` for any `ℕ`-graded ring `A`.
-/
def «Proj» : Scheme where
  __ := Proj.toLocallyRingedSpace 𝒜
  local_affine (x : Proj.T) := by
    classical
    obtain ⟨f, m, f_deg, hm, hx⟩ : ∃ (f : A) (m : ℕ) (_ : f ∈ 𝒜 m) (_ : 0 < m), f ∉ x.1 := by
      by_contra!
      refine x.not_irrelevant_le fun z hz ↦ ?_
      rw [← DirectSum.sum_support_decompose 𝒜 z]
      exact x.1.toIdeal.sum_mem fun k hk ↦ this _ k (SetLike.coe_mem _) <| by_contra <| by aesop
    exact ⟨⟨pbo f, hx⟩, .of (A⁰_ f), ⟨projIsoSpec 𝒜 f f_deg hm⟩⟩



