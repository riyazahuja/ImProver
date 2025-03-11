theorem Rat.RingOfIntegers.isUnit_iff {x : 𝓞 ℚ} : IsUnit x ↔ (x : ℚ) = 1 ∨ (x : ℚ) = -1 := by
  simp_rw [(isUnit_map_iff (Rat.ringOfIntegersEquiv : 𝓞 ℚ →+* ℤ) x).symm, Int.isUnit_iff,
    RingEquiv.coe_toRingHom, RingEquiv.map_eq_one_iff, RingEquiv.map_eq_neg_one_iff, ←
                                   /-
                                     x : NumberField.RingOfIntegers Rat
                                     ⊢ Iff (Or (Eq ↑x ↑1) (Eq ↑x ↑(-1))) (Or (Eq (↑x) 1) (Eq (↑x) (-1)))
                                   -/
    Subtype.coe_injective.eq_iff]; rfl
                                   /-
                                     🎉 no goals
                                   -/


theorem NumberField.isUnit_iff_norm [NumberField K] {x : 𝓞 K} :
    IsUnit x ↔ |(RingOfIntegers.norm ℚ x : ℚ)| = 1 := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    x : NumberField.RingOfIntegers K
    ⊢ Iff (IsUnit x) (Eq (abs ↑((RingOfIntegers.norm Rat) x)) 1)
  -/
  convert (RingOfIntegers.isUnit_norm ℚ (F := K)).symm
  /-
    case h.e'_2.a
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    x : NumberField.RingOfIntegers K
    ⊢ Iff (Eq (abs ↑((RingOfIntegers.norm Rat) x)) 1) (IsUnit ((RingOfIntegers.nor …
  -/
  rw [← abs_one, abs_eq_abs, ← Rat.RingOfIntegers.isUnit_iff]
  /-
    🎉 no goals
  -/


instance : CoeHTC (𝓞 K)ˣ K :=
  ⟨fun x => algebraMap _ K (Units.val x)⟩


theorem coe_injective : Function.Injective ((↑) : (𝓞 K)ˣ → K) :=
  RingOfIntegers.coe_injective.comp Units.ext


theorem coe_coe (u : (𝓞 K)ˣ) : ((u : 𝓞 K) : K) = (u : K) := rfl


theorem coe_mul (x y : (𝓞 K)ˣ) : ((x * y : (𝓞 K)ˣ) : K) = (x : K) * (y : K) := rfl


theorem coe_pow (x : (𝓞 K)ˣ) (n : ℕ) : ((x ^ n : (𝓞 K)ˣ) : K) = (x : K) ^ n := by
  /-
    K : Type u_1
    inst✝ : Field K
    x : Units (NumberField.RingOfIntegers K)
    n : Nat
    ⊢ Eq ((algebraMap (NumberField.RingOfIntegers K) K) ↑(HPow.hPow x n)) (HPow.hP …
  -/
  rw [← map_pow, ← val_pow_eq_pow_val]
  /-
    🎉 no goals
  -/


theorem coe_zpow (x : (𝓞 K)ˣ) (n : ℤ) : (↑(x ^ n) : K) = (x : K) ^ n := by
  /-
    K : Type u_1
    inst✝ : Field K
    x : Units (NumberField.RingOfIntegers K)
    n : Int
    ⊢ Eq ((algebraMap (NumberField.RingOfIntegers K) K) ↑(HPow.hPow x n)) (HPow.hP …
  -/
  change ((Units.coeHom K).comp (map (algebraMap (𝓞 K) K))) (x ^ n) = _
  /-
    K : Type u_1
    inst✝ : Field K
    x : Units (NumberField.RingOfIntegers K)
    n : Int
    ⊢ Eq (((Units.coeHom K).comp (Units.map ↑(algebraMap (NumberField.RingOfIntege …
  -/
  exact map_zpow _ x n
  /-
    🎉 no goals
  -/


theorem coe_one : ((1 : (𝓞 K)ˣ) : K) = (1 : K) := rfl


theorem coe_neg_one : ((-1 : (𝓞 K)ˣ) : K) = (-1 : K) := rfl


theorem coe_ne_zero (x : (𝓞 K)ˣ) : (x : K) ≠ 0 :=
  Subtype.coe_injective.ne_iff.mpr (_root_.Units.ne_zero x)


@[simp]
protected theorem norm [NumberField K] (x : (𝓞 K)ˣ) :
    |Algebra.norm ℚ (x : K)| = 1 := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    x : Units (NumberField.RingOfIntegers K)
    ⊢ Eq (abs ((Algebra.norm Rat) ((algebraMap (NumberField.RingOfIntegers K) K) ↑ …
  -/
  rw [← RingOfIntegers.coe_norm, isUnit_iff_norm.mp x.isUnit]
  /-
    🎉 no goals
  -/


/-- The torsion subgroup of the group of units. -/
def torsion : Subgroup (𝓞 K)ˣ := CommGroup.torsion (𝓞 K)ˣ


theorem mem_torsion {x : (𝓞 K)ˣ} [NumberField K] :
    x ∈ torsion K ↔ ∀ w : InfinitePlace K, w x = 1 := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    x : Units (NumberField.RingOfIntegers K)
    inst✝ : NumberField K
    ⊢ Iff (Membership.mem (NumberField.Units.torsion K) x) (∀ (w : NumberField.Inf …
  -/
  rw [eq_iff_eq (x : K) 1, torsion, CommGroup.mem_torsion]
  refine ⟨fun hx φ ↦ (((φ.comp <| algebraMap (𝓞 K) K).toMonoidHom.comp <|
    Units.coeHom _).isOfFinOrder hx).norm_eq_one, fun h ↦ isOfFinOrder_iff_pow_eq_one.2 ?_⟩
  /-
    K : Type u_1
    inst✝¹ : Field K
    x : Units (NumberField.RingOfIntegers K)
    inst✝ : NumberField K
    h : ∀ (φ : RingHom K Complex), Eq (Norm.norm (φ ((algebraMap (NumberField.Ring …
    ⊢ Exists fun n => And (LT.lt 0 n) (Eq (HPow.hPow x n) 1)
  -/
  obtain ⟨n, hn, hx⟩ := Embeddings.pow_eq_one_of_norm_eq_one K ℂ x.val.isIntegral_coe h
  exact ⟨n, hn, by ext; rw [NumberField.RingOfIntegers.coe_eq_algebraMap, coe_pow, hx,
    NumberField.RingOfIntegers.coe_eq_algebraMap, coe_one]⟩


/-- The torsion subgroup is finite. -/
instance [NumberField K] : Fintype (torsion K) := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    ⊢ Fintype (Subtype fun x => Membership.mem (NumberField.Units.torsion K) x)
  -/
  refine @Fintype.ofFinite _ (Set.finite_coe_iff.mpr ?_)
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    ⊢ (↑(NumberField.Units.torsion K)).Finite
  -/
  refine Set.Finite.of_finite_image ?_ (coe_injective K).injOn
  refine (Embeddings.finite_of_norm_le K ℂ 1).subset
    (fun a ⟨u, ⟨h_tors, h_ua⟩⟩ => ⟨?_, fun φ => ?_⟩)
    /-
      case refine_1
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      a : K
      x✝ : Membership.mem (Set.image (fun x => (algebraMap (NumberField.RingOfIntege …
      u : Units (NumberField.RingOfIntegers K)
      h_tors : Membership.mem (↑(NumberField.Units.torsion K)) u
      h_ua : Eq ((fun x => (algebraMap (NumberField.RingOfIntegers K) K) ↑x) u) a
      ⊢ IsIntegral Int a
    -/
  · rw [← h_ua]
    /-
      case refine_1
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      a : K
      x✝ : Membership.mem (Set.image (fun x => (algebraMap (NumberField.RingOfIntege …
      u : Units (NumberField.RingOfIntegers K)
      h_tors : Membership.mem (↑(NumberField.Units.torsion K)) u
      h_ua : Eq ((fun x => (algebraMap (NumberField.RingOfIntegers K) K) ↑x) u) a
      ⊢ IsIntegral Int ((fun x => (algebraMap (NumberField.RingOfIntegers K) K) ↑x) u)
    -/
    exact u.val.prop
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      a : K
      x✝ : Membership.mem (Set.image (fun x => (algebraMap (NumberField.RingOfIntege …
      u : Units (NumberField.RingOfIntegers K)
      h_tors : Membership.mem (↑(NumberField.Units.torsion K)) u
      h_ua : Eq ((fun x => (algebraMap (NumberField.RingOfIntegers K) K) ↑x) u) a
      φ : RingHom K Complex
      ⊢ LE.le (Norm.norm (φ a)) 1
    -/
  · rw [← h_ua]
    /-
      case refine_2
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      a : K
      x✝ : Membership.mem (Set.image (fun x => (algebraMap (NumberField.RingOfIntege …
      u : Units (NumberField.RingOfIntegers K)
      h_tors : Membership.mem (↑(NumberField.Units.torsion K)) u
      h_ua : Eq ((fun x => (algebraMap (NumberField.RingOfIntegers K) K) ↑x) u) a
      φ : RingHom K Complex
      ⊢ LE.le (Norm.norm (φ ((fun x => (algebraMap (NumberField.RingOfIntegers K) K) …
    -/
    exact le_of_eq ((eq_iff_eq _ 1).mp ((mem_torsion K).mp h_tors) φ)
    /-
      🎉 no goals
    -/


instance : Nonempty (torsion K) := One.instNonempty


/-- The torsion subgroup is cyclic. -/
instance [NumberField K] : IsCyclic (torsion K) := subgroup_units_cyclic _


/-- The order of the torsion subgroup as a positive integer. -/
def torsionOrder [NumberField K] : ℕ+ := ⟨Fintype.card (torsion K), Fintype.card_pos⟩


/-- If `k` does not divide `torsionOrder` then there are no nontrivial roots of unity of
  order dividing `k`. -/
theorem rootsOfUnity_eq_one [NumberField K] {k : ℕ+} (hc : Nat.Coprime k (torsionOrder K))
    {ζ : (𝓞 K)ˣ} : ζ ∈ rootsOfUnity k (𝓞 K) ↔ ζ = 1 := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    k : PNat
    hc : (↑k).Coprime ↑(NumberField.Units.torsionOrder K)
    ζ : Units (NumberField.RingOfIntegers K)
    ⊢ Iff (Membership.mem (rootsOfUnity (↑k) (NumberField.RingOfIntegers K)) ζ) (E …
  -/
  rw [mem_rootsOfUnity]
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    k : PNat
    hc : (↑k).Coprime ↑(NumberField.Units.torsionOrder K)
    ζ : Units (NumberField.RingOfIntegers K)
    ⊢ Iff (Eq (HPow.hPow ζ ↑k) 1) (Eq ζ 1)
  -/
  refine ⟨fun h => ?_, fun h => by rw [h, one_pow]⟩
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    k : PNat
    hc : (↑k).Coprime ↑(NumberField.Units.torsionOrder K)
    ζ : Units (NumberField.RingOfIntegers K)
    h : Eq (HPow.hPow ζ ↑k) 1
    ⊢ Eq ζ 1
  -/
  refine orderOf_eq_one_iff.mp (Nat.eq_one_of_dvd_coprimes hc ?_ ?_)
    /-
      case refine_1
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      k : PNat
      hc : (↑k).Coprime ↑(NumberField.Units.torsionOrder K)
      ζ : Units (NumberField.RingOfIntegers K)
      h : Eq (HPow.hPow ζ ↑k) 1
      ⊢ Dvd.dvd (orderOf ζ) ↑k
    -/
  · exact orderOf_dvd_of_pow_eq_one h
    /-
      🎉 no goals
    -/
  · have hζ : ζ ∈ torsion K := by
      rw [torsion, CommGroup.mem_torsion, isOfFinOrder_iff_pow_eq_one]
      exact ⟨k, k.prop, h⟩
    /-
      case refine_2
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      k : PNat
      hc : (↑k).Coprime ↑(NumberField.Units.torsionOrder K)
      ζ : Units (NumberField.RingOfIntegers K)
      h : Eq (HPow.hPow ζ ↑k) 1
      hζ : Membership.mem (NumberField.Units.torsion K) ζ
      ⊢ Dvd.dvd (orderOf ζ) ↑(NumberField.Units.torsionOrder K)
    -/
    rw [orderOf_submonoid (⟨ζ, hζ⟩ : torsion K)]
    /-
      case refine_2
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      k : PNat
      hc : (↑k).Coprime ↑(NumberField.Units.torsionOrder K)
      ζ : Units (NumberField.RingOfIntegers K)
      h : Eq (HPow.hPow ζ ↑k) 1
      hζ : Membership.mem (NumberField.Units.torsion K) ζ
      ⊢ Dvd.dvd (orderOf ⟨ζ, hζ⟩) ↑(NumberField.Units.torsionOrder K)
    -/
    exact orderOf_dvd_card
    /-
      🎉 no goals
    -/


/-- The group of roots of unity of order dividing `torsionOrder` is equal to the torsion
group. -/
theorem rootsOfUnity_eq_torsion [NumberField K] :
    rootsOfUnity (torsionOrder K) (𝓞 K) = torsion K := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    ⊢ Eq (rootsOfUnity (↑(NumberField.Units.torsionOrder K)) (NumberField.RingOfIn …
  -/
  ext ζ
  /-
    case h
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    ζ : Units (NumberField.RingOfIntegers K)
    ⊢ Iff (Membership.mem (rootsOfUnity (↑(NumberField.Units.torsionOrder K)) (Num …
  -/
  rw [torsion, mem_rootsOfUnity]
  /-
    case h
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    ζ : Units (NumberField.RingOfIntegers K)
    ⊢ Iff (Eq (HPow.hPow ζ ↑(NumberField.Units.torsionOrder K)) 1) (Membership.mem …
  -/
  refine ⟨fun h => ?_, fun h => ?_⟩
    /-
      case h.refine_1
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      ζ : Units (NumberField.RingOfIntegers K)
      h : Eq (HPow.hPow ζ ↑(NumberField.Units.torsionOrder K)) 1
      ⊢ Membership.mem (CommGroup.torsion (Units (NumberField.RingOfIntegers K))) ζ
    -/
  · rw [CommGroup.mem_torsion, isOfFinOrder_iff_pow_eq_one]
    /-
      case h.refine_1
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      ζ : Units (NumberField.RingOfIntegers K)
      h : Eq (HPow.hPow ζ ↑(NumberField.Units.torsionOrder K)) 1
      ⊢ Exists fun n => And (LT.lt 0 n) (Eq (HPow.hPow ζ n) 1)
    -/
    exact ⟨↑(torsionOrder K), (torsionOrder K).prop, h⟩
    /-
      🎉 no goals
    -/
    /-
      case h.refine_2
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      ζ : Units (NumberField.RingOfIntegers K)
      h : Membership.mem (CommGroup.torsion (Units (NumberField.RingOfIntegers K))) ζ
      ⊢ Eq (HPow.hPow ζ ↑(NumberField.Units.torsionOrder K)) 1
    -/
  · exact Subtype.ext_iff.mp (@pow_card_eq_one (torsion K) _ _ ⟨ζ, h⟩)
    /-
      🎉 no goals
    -/


