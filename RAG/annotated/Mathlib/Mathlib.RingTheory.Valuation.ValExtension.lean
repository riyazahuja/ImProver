/--
The class `IsValExtension R A` states that the valuation of `A` is an extension of the valuation
on `R`. More precisely, the valuation on `R` is equivalent to the comap of the valuation on `A`.
-/
class IsValExtension : Prop where
  /-- The valuation on `R` is equivalent to the comap of the valuation on `A` -/
  val_isEquiv_comap : vR.IsEquiv <| vA.comap (algebraMap R A)


theorem val_map_le_iff (x y : R) : vA (algebraMap R A x) ≤ vA (algebraMap R A y) ↔ vR x ≤ vR y :=
  val_isEquiv_comap.symm x y


theorem val_map_lt_iff (x y : R) : vA (algebraMap R A x) < vA (algebraMap R A y) ↔ vR x < vR y := by
  /-
    R : Type u_1
    A : Type u_2
    ΓR : Type u_3
    ΓA : Type u_4
    inst✝⁵ : CommRing R
    inst✝⁴ : Ring A
    inst✝³ : LinearOrderedCommMonoidWithZero ΓR
    inst✝² : LinearOrderedCommMonoidWithZero ΓA
    inst✝¹ : Algebra R A
    vR : Valuation R ΓR
    vA : Valuation A ΓA
    inst✝ : IsValExtension vR vA
    x y : R
    ⊢ Iff (LT.lt (vA ((algebraMap R A) x)) (vA ((algebraMap R A) y))) (LT.lt (vR x …
  -/
  simpa only [not_le] using ((val_map_le_iff vR vA _ _).not)
  /-
    🎉 no goals
  -/


theorem val_map_eq_iff (x y : R) : vA (algebraMap R A x) = vA (algebraMap R A y) ↔ vR x = vR y :=
  (IsEquiv.val_eq val_isEquiv_comap).symm


theorem val_map_le_one_iff (x : R) : vA (algebraMap R A x) ≤ 1 ↔ vR x ≤ 1 := by
  /-
    R : Type u_1
    A : Type u_2
    ΓR : Type u_3
    ΓA : Type u_4
    inst✝⁵ : CommRing R
    inst✝⁴ : Ring A
    inst✝³ : LinearOrderedCommMonoidWithZero ΓR
    inst✝² : LinearOrderedCommMonoidWithZero ΓA
    inst✝¹ : Algebra R A
    vR : Valuation R ΓR
    vA : Valuation A ΓA
    inst✝ : IsValExtension vR vA
    x : R
    ⊢ Iff (LE.le (vA ((algebraMap R A) x)) 1) (LE.le (vR x) 1)
  -/
  simpa only [_root_.map_one] using val_map_le_iff vR vA x 1
  /-
    🎉 no goals
  -/


theorem val_map_lt_one_iff (x : R) : vA (algebraMap R A x) < 1 ↔ vR x < 1 := by
  /-
    R : Type u_1
    A : Type u_2
    ΓR : Type u_3
    ΓA : Type u_4
    inst✝⁵ : CommRing R
    inst✝⁴ : Ring A
    inst✝³ : LinearOrderedCommMonoidWithZero ΓR
    inst✝² : LinearOrderedCommMonoidWithZero ΓA
    inst✝¹ : Algebra R A
    vR : Valuation R ΓR
    vA : Valuation A ΓA
    inst✝ : IsValExtension vR vA
    x : R
    ⊢ Iff (LT.lt (vA ((algebraMap R A) x)) 1) (LT.lt (vR x) 1)
  -/
  simpa only [_root_.map_one, not_le] using (val_map_le_iff vR vA 1 x).not
  /-
    🎉 no goals
  -/


theorem val_map_eq_one_iff (x : R) : vA (algebraMap R A x) = 1 ↔ vR x = 1 := by
  simpa only [le_antisymm_iff, _root_.map_one] using
    and_congr (val_map_le_iff vR vA x 1) (val_map_le_iff vR vA 1 x)


instance id : IsValExtension vR vR where
  val_isEquiv_comap := by
    /-
      R : Type u_1
      A : Type u_2
      ΓR : Type u_3
      ΓA : Type u_4
      inst✝⁴ : CommRing R
      inst✝³ : Ring A
      inst✝² : LinearOrderedCommMonoidWithZero ΓR
      inst✝¹ : LinearOrderedCommMonoidWithZero ΓA
      inst✝ : Algebra R A
      vR : Valuation R ΓR
      vA : Valuation A ΓA
      ⊢ vR.IsEquiv (Valuation.comap (algebraMap R R) vR)
    -/
    simp only [Algebra.id.map_eq_id, comap_id, IsEquiv.refl]
    /-
      🎉 no goals
    -/


/--
When `K` is a field, if the preimage of the valuation integers of `A` equals to the valuation
integers of `K`, then the valuation on `A` is an extension of the valuation on `K`.
-/
theorem ofComapInteger (h : vA.integer.comap (algebraMap K A) = vK.integer) :
    IsValExtension vK vA where
  val_isEquiv_comap := by
    /-
      A : Type u_2
      inst✝⁴ : Ring A
      K : Type u_5
      inst✝³ : Field K
      inst✝² : Algebra K A
      ΓA : Type u_7
      ΓK : Type u_8
      inst✝¹ : LinearOrderedCommGroupWithZero ΓK
      inst✝ : LinearOrderedCommGroupWithZero ΓA
      vK : Valuation K ΓK
      vA : Valuation A ΓA
      h : Eq (Subring.comap (algebraMap K A) vA.integer) vK.integer
      ⊢ vK.IsEquiv (Valuation.comap (algebraMap K A) vA)
    -/
    rw [isEquiv_iff_val_le_one]
    /-
      A : Type u_2
      inst✝⁴ : Ring A
      K : Type u_5
      inst✝³ : Field K
      inst✝² : Algebra K A
      ΓA : Type u_7
      ΓK : Type u_8
      inst✝¹ : LinearOrderedCommGroupWithZero ΓK
      inst✝ : LinearOrderedCommGroupWithZero ΓA
      vK : Valuation K ΓK
      vA : Valuation A ΓA
      h : Eq (Subring.comap (algebraMap K A) vA.integer) vK.integer
      ⊢ ∀ {x : K}, Iff (LE.le (vK x) 1) (LE.le ((Valuation.comap (algebraMap K A) vA …
    -/
    intro x
    /-
      A : Type u_2
      inst✝⁴ : Ring A
      K : Type u_5
      inst✝³ : Field K
      inst✝² : Algebra K A
      ΓA : Type u_7
      ΓK : Type u_8
      inst✝¹ : LinearOrderedCommGroupWithZero ΓK
      inst✝ : LinearOrderedCommGroupWithZero ΓA
      vK : Valuation K ΓK
      vA : Valuation A ΓA
      h : Eq (Subring.comap (algebraMap K A) vA.integer) vK.integer
      x : K
      ⊢ Iff (LE.le (vK x) 1) (LE.le ((Valuation.comap (algebraMap K A) vA) x) 1)
    -/
    simp_rw [← Valuation.mem_integer_iff, ← h, Subring.mem_comap, mem_integer_iff, comap_apply]
    /-
      🎉 no goals
    -/


instance instAlgebraInteger : Algebra vR.integer vA.integer where
  smul r a := ⟨r • a,
    Algebra.smul_def r (a : A) ▸ mul_mem ((val_map_le_one_iff vR vA _).mpr r.2) a.2⟩
  __ := (algebraMap R A).restrict vR.integer vA.integer
        /-
          R : Type u_1
          A : Type u_2
          ΓR✝ : Type u_3
          ΓA✝ : Type u_4
          inst✝¹⁰ : CommRing R
          inst✝⁹ : Ring A
          inst✝⁸ : LinearOrderedCommMonoidWithZero ΓR✝
          inst✝⁷ : LinearOrderedCommMonoidWithZero ΓA✝
          inst✝⁶ : Algebra R A
          vR✝ : Valuation R ΓR✝
          vA✝ : Valuation A ΓA✝
          K : Type u_5
          inst✝⁵ : Field K
          inst✝⁴ : Algebra K A
          ΓR : Type u_6
          ΓA : Type u_7
          ΓK : Type u_8
          inst✝³ : LinearOrderedCommGroupWithZero ΓR
          inst✝² : LinearOrderedCommGroupWithZero ΓK
          inst✝¹ : LinearOrderedCommGroupWithZero ΓA
          vR : Valuation R ΓR
          vK : Valuation K ΓK
          vA : Valuation A ΓA
          inst✝ : IsValExtension vR vA
          ⊢ ∀ (x : R), Membership.mem vR.integer x → Membership.mem vA.integer ((algebra …
        -/
    (by simp [Valuation.mem_integer_iff, val_map_le_one_iff vR vA])
        /-
          🎉 no goals
        -/
  commutes' _ _ := Subtype.ext (Algebra.commutes _ _)
  smul_def' _ _ := Subtype.ext (Algebra.smul_def _ _)


@[simp, norm_cast]
theorem val_smul (r : vR.integer) (a : vA.integer) : ↑(r • a : vA.integer) = (r : R) • (a : A) := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝⁵ : CommRing R
    inst✝⁴ : Ring A
    inst✝³ : Algebra R A
    ΓR : Type u_6
    ΓA : Type u_7
    inst✝² : LinearOrderedCommGroupWithZero ΓR
    inst✝¹ : LinearOrderedCommGroupWithZero ΓA
    vR : Valuation R ΓR
    vA : Valuation A ΓA
    inst✝ : IsValExtension vR vA
    r : Subtype fun x => Membership.mem vR.integer x
    a : Subtype fun x => Membership.mem vA.integer x
    ⊢ Eq (↑(HSMul.hSMul r a)) (HSMul.hSMul ↑r ↑a)
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
theorem val_algebraMap (r : vR.integer) :
    ((algebraMap vR.integer vA.integer) r : A) = (algebraMap R A) (r : R) := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝⁵ : CommRing R
    inst✝⁴ : Ring A
    inst✝³ : Algebra R A
    ΓR : Type u_6
    ΓA : Type u_7
    inst✝² : LinearOrderedCommGroupWithZero ΓR
    inst✝¹ : LinearOrderedCommGroupWithZero ΓA
    vR : Valuation R ΓR
    vA : Valuation A ΓA
    inst✝ : IsValExtension vR vA
    r : Subtype fun x => Membership.mem vR.integer x
    ⊢ Eq (↑((algebraMap (Subtype fun x => Membership.mem vR.integer x) (Subtype fu …
  -/
  rfl
  /-
    🎉 no goals
  -/


instance instIsScalarTowerInteger : IsScalarTower vR.integer vA.integer A where
  smul_assoc x y z := by
    /-
      R : Type u_1
      A : Type u_2
      ΓR✝ : Type u_3
      ΓA✝ : Type u_4
      inst✝¹⁰ : CommRing R
      inst✝⁹ : Ring A
      inst✝⁸ : LinearOrderedCommMonoidWithZero ΓR✝
      inst✝⁷ : LinearOrderedCommMonoidWithZero ΓA✝
      inst✝⁶ : Algebra R A
      vR✝ : Valuation R ΓR✝
      vA✝ : Valuation A ΓA✝
      K : Type u_5
      inst✝⁵ : Field K
      inst✝⁴ : Algebra K A
      ΓR : Type u_6
      ΓA : Type u_7
      ΓK : Type u_8
      inst✝³ : LinearOrderedCommGroupWithZero ΓR
      inst✝² : LinearOrderedCommGroupWithZero ΓK
      inst✝¹ : LinearOrderedCommGroupWithZero ΓA
      vR : Valuation R ΓR
      vK : Valuation K ΓK
      vA : Valuation A ΓA
      inst✝ : IsValExtension vR vA
      x : Subtype fun x => Membership.mem vR.integer x
      y : Subtype fun x => Membership.mem vA.integer x
      z : A
      ⊢ Eq (HSMul.hSMul (HSMul.hSMul x y) z) (HSMul.hSMul x (HSMul.hSMul y z))
    -/
    simp only [Algebra.smul_def]
    /-
      R : Type u_1
      A : Type u_2
      ΓR✝ : Type u_3
      ΓA✝ : Type u_4
      inst✝¹⁰ : CommRing R
      inst✝⁹ : Ring A
      inst✝⁸ : LinearOrderedCommMonoidWithZero ΓR✝
      inst✝⁷ : LinearOrderedCommMonoidWithZero ΓA✝
      inst✝⁶ : Algebra R A
      vR✝ : Valuation R ΓR✝
      vA✝ : Valuation A ΓA✝
      K : Type u_5
      inst✝⁵ : Field K
      inst✝⁴ : Algebra K A
      ΓR : Type u_6
      ΓA : Type u_7
      ΓK : Type u_8
      inst✝³ : LinearOrderedCommGroupWithZero ΓR
      inst✝² : LinearOrderedCommGroupWithZero ΓK
      inst✝¹ : LinearOrderedCommGroupWithZero ΓA
      vR : Valuation R ΓR
      vK : Valuation K ΓK
      vA : Valuation A ΓA
      inst✝ : IsValExtension vR vA
      x : Subtype fun x => Membership.mem vR.integer x
      y : Subtype fun x => Membership.mem vA.integer x
      z : A
      ⊢ Eq (HSMul.hSMul (HMul.hMul ((algebraMap (Subtype fun x => Membership.mem vR. …
    -/
    exact mul_assoc _ _ _
    /-
      🎉 no goals
    -/


instance instNoZeroSMulDivisorsInteger [NoZeroSMulDivisors R A] :
    NoZeroSMulDivisors vR.integer vA.integer := by
  /-
    R : Type u_1
    A : Type u_2
    ΓR✝ : Type u_3
    ΓA✝ : Type u_4
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : Ring A
    inst✝⁹ : LinearOrderedCommMonoidWithZero ΓR✝
    inst✝⁸ : LinearOrderedCommMonoidWithZero ΓA✝
    inst✝⁷ : Algebra R A
    vR✝ : Valuation R ΓR✝
    vA✝ : Valuation A ΓA✝
    K : Type u_5
    inst✝⁶ : Field K
    inst✝⁵ : Algebra K A
    ΓR : Type u_6
    ΓA : Type u_7
    ΓK : Type u_8
    inst✝⁴ : LinearOrderedCommGroupWithZero ΓR
    inst✝³ : LinearOrderedCommGroupWithZero ΓK
    inst✝² : LinearOrderedCommGroupWithZero ΓA
    vR : Valuation R ΓR
    vK : Valuation K ΓK
    vA : Valuation A ΓA
    inst✝¹ : IsValExtension vR vA
    inst✝ : NoZeroSMulDivisors R A
    ⊢ NoZeroSMulDivisors (Subtype fun x => Membership.mem vR.integer x) (Subtype f …
  -/
  refine ⟨fun {x y} e ↦ ?_⟩
  /-
    R : Type u_1
    A : Type u_2
    ΓR✝ : Type u_3
    ΓA✝ : Type u_4
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : Ring A
    inst✝⁹ : LinearOrderedCommMonoidWithZero ΓR✝
    inst✝⁸ : LinearOrderedCommMonoidWithZero ΓA✝
    inst✝⁷ : Algebra R A
    vR✝ : Valuation R ΓR✝
    vA✝ : Valuation A ΓA✝
    K : Type u_5
    inst✝⁶ : Field K
    inst✝⁵ : Algebra K A
    ΓR : Type u_6
    ΓA : Type u_7
    ΓK : Type u_8
    inst✝⁴ : LinearOrderedCommGroupWithZero ΓR
    inst✝³ : LinearOrderedCommGroupWithZero ΓK
    inst✝² : LinearOrderedCommGroupWithZero ΓA
    vR : Valuation R ΓR
    vK : Valuation K ΓK
    vA : Valuation A ΓA
    inst✝¹ : IsValExtension vR vA
    inst✝ : NoZeroSMulDivisors R A
    x : Subtype fun x => Membership.mem vR.integer x
    y : Subtype fun x => Membership.mem vA.integer x
    e : Eq (HSMul.hSMul x y) 0
    ⊢ Or (Eq x 0) (Eq y 0)
  -/
  have : (x : R) • (y : A) = 0 := by simpa [Subtype.ext_iff, Algebra.smul_def] using e
  /-
    R : Type u_1
    A : Type u_2
    ΓR✝ : Type u_3
    ΓA✝ : Type u_4
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : Ring A
    inst✝⁹ : LinearOrderedCommMonoidWithZero ΓR✝
    inst✝⁸ : LinearOrderedCommMonoidWithZero ΓA✝
    inst✝⁷ : Algebra R A
    vR✝ : Valuation R ΓR✝
    vA✝ : Valuation A ΓA✝
    K : Type u_5
    inst✝⁶ : Field K
    inst✝⁵ : Algebra K A
    ΓR : Type u_6
    ΓA : Type u_7
    ΓK : Type u_8
    inst✝⁴ : LinearOrderedCommGroupWithZero ΓR
    inst✝³ : LinearOrderedCommGroupWithZero ΓK
    inst✝² : LinearOrderedCommGroupWithZero ΓA
    vR : Valuation R ΓR
    vK : Valuation K ΓK
    vA : Valuation A ΓA
    inst✝¹ : IsValExtension vR vA
    inst✝ : NoZeroSMulDivisors R A
    x : Subtype fun x => Membership.mem vR.integer x
    y : Subtype fun x => Membership.mem vA.integer x
    e : Eq (HSMul.hSMul x y) 0
    this : Eq (HSMul.hSMul ↑x ↑y) 0
    ⊢ Or (Eq x 0) (Eq y 0)
  -/
  simpa only [Subtype.ext_iff, smul_eq_zero] using this
  /-
    🎉 no goals
  -/


theorem algebraMap_injective [IsValExtension vK vA] [Nontrivial A] :
    Function.Injective (algebraMap vK.integer vA.integer) := by
  /-
    A : Type u_2
    inst✝⁶ : Ring A
    K : Type u_5
    inst✝⁵ : Field K
    inst✝⁴ : Algebra K A
    ΓA : Type u_7
    ΓK : Type u_8
    inst✝³ : LinearOrderedCommGroupWithZero ΓK
    inst✝² : LinearOrderedCommGroupWithZero ΓA
    vK : Valuation K ΓK
    vA : Valuation A ΓA
    inst✝¹ : IsValExtension vK vA
    inst✝ : Nontrivial A
    ⊢ Function.Injective ⇑(algebraMap (Subtype fun x => Membership.mem vK.integer  …
  -/
  intro x y h
  /-
    A : Type u_2
    inst✝⁶ : Ring A
    K : Type u_5
    inst✝⁵ : Field K
    inst✝⁴ : Algebra K A
    ΓA : Type u_7
    ΓK : Type u_8
    inst✝³ : LinearOrderedCommGroupWithZero ΓK
    inst✝² : LinearOrderedCommGroupWithZero ΓA
    vK : Valuation K ΓK
    vA : Valuation A ΓA
    inst✝¹ : IsValExtension vK vA
    inst✝ : Nontrivial A
    x y : Subtype fun x => Membership.mem vK.integer x
    h : Eq ((algebraMap (Subtype fun x => Membership.mem vK.integer x) (Subtype fu …
    ⊢ Eq x y
  -/
  simp only [Subtype.ext_iff, val_algebraMap] at h
  /-
    A : Type u_2
    inst✝⁶ : Ring A
    K : Type u_5
    inst✝⁵ : Field K
    inst✝⁴ : Algebra K A
    ΓA : Type u_7
    ΓK : Type u_8
    inst✝³ : LinearOrderedCommGroupWithZero ΓK
    inst✝² : LinearOrderedCommGroupWithZero ΓA
    vK : Valuation K ΓK
    vA : Valuation A ΓA
    inst✝¹ : IsValExtension vK vA
    inst✝ : Nontrivial A
    x y : Subtype fun x => Membership.mem vK.integer x
    h : Eq ((algebraMap K A) ↑x) ((algebraMap K A) ↑y)
    ⊢ Eq x y
  -/
  ext
  /-
    case a
    A : Type u_2
    inst✝⁶ : Ring A
    K : Type u_5
    inst✝⁵ : Field K
    inst✝⁴ : Algebra K A
    ΓA : Type u_7
    ΓK : Type u_8
    inst✝³ : LinearOrderedCommGroupWithZero ΓK
    inst✝² : LinearOrderedCommGroupWithZero ΓA
    vK : Valuation K ΓK
    vA : Valuation A ΓA
    inst✝¹ : IsValExtension vK vA
    inst✝ : Nontrivial A
    x y : Subtype fun x => Membership.mem vK.integer x
    h : Eq ((algebraMap K A) ↑x) ((algebraMap K A) ↑y)
    ⊢ Eq ↑x ↑y
  -/
  apply RingHom.injective (algebraMap K A) h
  /-
    🎉 no goals
  -/


@[instance]
theorem instIsLocalHomValuationInteger {S ΓS: Type*} [CommRing S]
    [LinearOrderedCommGroupWithZero ΓS]
    [Algebra R S] [IsLocalHom (algebraMap R S)] {vS : Valuation S ΓS}
    [IsValExtension vR vS] : IsLocalHom (algebraMap vR.integer vS.integer) where
  map_nonunit r hr := by
    /-
      R : Type u_1
      inst✝⁶ : CommRing R
      ΓR : Type u_6
      inst✝⁵ : LinearOrderedCommGroupWithZero ΓR
      vR : Valuation R ΓR
      S : Type u_9
      ΓS : Type u_10
      inst✝⁴ : CommRing S
      inst✝³ : LinearOrderedCommGroupWithZero ΓS
      inst✝² : Algebra R S
      inst✝¹ : IsLocalHom (algebraMap R S)
      vS : Valuation S ΓS
      inst✝ : IsValExtension vR vS
      r : Subtype fun x => Membership.mem vR.integer x
      hr : IsUnit ((algebraMap (Subtype fun x => Membership.mem vR.integer x) (Subty …
      ⊢ IsUnit r
    -/
    apply (Valuation.integer.integers (v := vR)).isUnit_of_one
      /-
        case hx
        R : Type u_1
        inst✝⁶ : CommRing R
        ΓR : Type u_6
        inst✝⁵ : LinearOrderedCommGroupWithZero ΓR
        vR : Valuation R ΓR
        S : Type u_9
        ΓS : Type u_10
        inst✝⁴ : CommRing S
        inst✝³ : LinearOrderedCommGroupWithZero ΓS
        inst✝² : Algebra R S
        inst✝¹ : IsLocalHom (algebraMap R S)
        vS : Valuation S ΓS
        inst✝ : IsValExtension vR vS
        r : Subtype fun x => Membership.mem vR.integer x
        hr : IsUnit ((algebraMap (Subtype fun x => Membership.mem vR.integer x) (Subty …
        ⊢ IsUnit ((algebraMap (Subtype fun x => Membership.mem vR.integer x) R) r)
      -/
    · exact (isUnit_map_iff (algebraMap R S) _).mp (hr.map (algebraMap _ S))
      /-
        🎉 no goals
      -/
      /-
        case hvx
        R : Type u_1
        inst✝⁶ : CommRing R
        ΓR : Type u_6
        inst✝⁵ : LinearOrderedCommGroupWithZero ΓR
        vR : Valuation R ΓR
        S : Type u_9
        ΓS : Type u_10
        inst✝⁴ : CommRing S
        inst✝³ : LinearOrderedCommGroupWithZero ΓS
        inst✝² : Algebra R S
        inst✝¹ : IsLocalHom (algebraMap R S)
        vS : Valuation S ΓS
        inst✝ : IsValExtension vR vS
        r : Subtype fun x => Membership.mem vR.integer x
        hr : IsUnit ((algebraMap (Subtype fun x => Membership.mem vR.integer x) (Subty …
        ⊢ Eq (vR ((algebraMap (Subtype fun x => Membership.mem vR.integer x) R) r)) 1
      -/
    · apply (Valuation.integer.integers (v := vS)).one_of_isUnit at hr
      /-
        case hvx
        R : Type u_1
        inst✝⁶ : CommRing R
        ΓR : Type u_6
        inst✝⁵ : LinearOrderedCommGroupWithZero ΓR
        vR : Valuation R ΓR
        S : Type u_9
        ΓS : Type u_10
        inst✝⁴ : CommRing S
        inst✝³ : LinearOrderedCommGroupWithZero ΓS
        inst✝² : Algebra R S
        inst✝¹ : IsLocalHom (algebraMap R S)
        vS : Valuation S ΓS
        inst✝ : IsValExtension vR vS
        r : Subtype fun x => Membership.mem vR.integer x
        hr : Eq (vS ((algebraMap (Subtype fun x => Membership.mem vS.integer x) S) ((a …
        ⊢ Eq (vR ((algebraMap (Subtype fun x => Membership.mem vR.integer x) R) r)) 1
      -/
      exact (val_map_eq_one_iff vR vS _).mp hr
      /-
        🎉 no goals
      -/


@[deprecated (since := "2024-10-10")]
alias instIsLocalRingHomValuationInteger := instIsLocalHomValuationInteger


