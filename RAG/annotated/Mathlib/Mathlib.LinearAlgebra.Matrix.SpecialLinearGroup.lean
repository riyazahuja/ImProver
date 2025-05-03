/-- `SpecialLinearGroup n R` is the group of `n` by `n` `R`-matrices with determinant equal to 1.
-/
def SpecialLinearGroup :=
  { A : Matrix n n R // A.det = 1 }


@[inherit_doc]
scoped[MatrixGroups] notation "SL(" n ", " R ")" => Matrix.SpecialLinearGroup (Fin n) R


instance hasCoeToMatrix : Coe (SpecialLinearGroup n R) (Matrix n n R) :=
  ⟨fun A => A.val⟩


/-- In this file, Lean often has a hard time working out the values of `n` and `R` for an expression
like `det ↑A`. Rather than writing `(A : Matrix n n R)` everywhere in this file which is annoyingly
verbose, or `A.val` which is not the simp-normal form for subtypes, we create a local notation
`↑ₘA`. This notation references the local `n` and `R` variables, so is not valid as a global
notation. -/
local notation:1024 "↑ₘ" A:1024 => ((A : SpecialLinearGroup n R) : Matrix n n R)

-- Porting note: moved this section upwards because it used to be not simp-normal.
-- Now it is, since coercion arrows are unfolded.

/-- This instance is here for convenience, but is literally the same as the coercion from
`hasCoeToMatrix`. -/
instance instCoeFun : CoeFun (SpecialLinearGroup n R) fun _ => n → n → R where coe A := ↑ₘA


theorem ext_iff (A B : SpecialLinearGroup n R) : A = B ↔ ∀ i j, A i j = B i j :=
  Subtype.ext_iff.trans Matrix.ext_iff.symm


@[ext]
theorem ext (A B : SpecialLinearGroup n R) : (∀ i j, A i j = B i j) → A = B :=
  (SpecialLinearGroup.ext_iff A B).mpr


instance subsingleton_of_subsingleton [Subsingleton n] : Subsingleton (SpecialLinearGroup n R) := by
  /-
    n : Type u
    inst✝³ : DecidableEq n
    inst✝² : Fintype n
    R : Type v
    inst✝¹ : CommRing R
    inst✝ : Subsingleton n
    ⊢ Subsingleton (Matrix.SpecialLinearGroup n R)
  -/
  refine ⟨fun ⟨A, hA⟩ ⟨B, hB⟩ ↦ ?_⟩
  /-
    n : Type u
    inst✝³ : DecidableEq n
    inst✝² : Fintype n
    R : Type v
    inst✝¹ : CommRing R
    inst✝ : Subsingleton n
    x✝¹ x✝ : Matrix.SpecialLinearGroup n R
    A : Matrix n n R
    hA : Eq A.det 1
    B : Matrix n n R
    hB : Eq B.det 1
    ⊢ Eq ⟨A, hA⟩ ⟨B, hB⟩
  -/
  ext i j
  /-
    case a
    n : Type u
    inst✝³ : DecidableEq n
    inst✝² : Fintype n
    R : Type v
    inst✝¹ : CommRing R
    inst✝ : Subsingleton n
    x✝¹ x✝ : Matrix.SpecialLinearGroup n R
    A : Matrix n n R
    hA : Eq A.det 1
    B : Matrix n n R
    hB : Eq B.det 1
    i j : n
    ⊢ Eq (↑⟨A, hA⟩ i j) (↑⟨B, hB⟩ i j)
  -/
  rcases isEmpty_or_nonempty n with hn | hn; · exfalso; exact IsEmpty.false i
                                                        /-
                                                          🎉 no goals
                                                        -/
  /-
    case a.inr
    n : Type u
    inst✝³ : DecidableEq n
    inst✝² : Fintype n
    R : Type v
    inst✝¹ : CommRing R
    inst✝ : Subsingleton n
    x✝¹ x✝ : Matrix.SpecialLinearGroup n R
    A : Matrix n n R
    hA : Eq A.det 1
    B : Matrix n n R
    hB : Eq B.det 1
    i j : n
    hn : Nonempty n
    ⊢ Eq (↑⟨A, hA⟩ i j) (↑⟨B, hB⟩ i j)
  -/
  rw [det_eq_elem_of_subsingleton _ i] at hA hB
  /-
    case a.inr
    n : Type u
    inst✝³ : DecidableEq n
    inst✝² : Fintype n
    R : Type v
    inst✝¹ : CommRing R
    inst✝ : Subsingleton n
    x✝¹ x✝ : Matrix.SpecialLinearGroup n R
    A : Matrix n n R
    hA✝ : Eq A.det 1
    B : Matrix n n R
    hB✝ : Eq B.det 1
    i : n
    hB : Eq (B i i) 1
    hA : Eq (A i i) 1
    j : n
    hn : Nonempty n
    ⊢ Eq (↑⟨A, hA✝⟩ i j) (↑⟨B, hB✝⟩ i j)
  -/
  simp only [Subsingleton.elim j i, hA, hB]
  /-
    🎉 no goals
  -/


instance hasInv : Inv (SpecialLinearGroup n R) :=
                            /-
                              n : Type u
                              inst✝² : DecidableEq n
                              inst✝¹ : Fintype n
                              R : Type v
                              inst✝ : CommRing R
                              A : Matrix.SpecialLinearGroup n R
                              ⊢ Eq (↑A).adjugate.det 1
                            -/
  ⟨fun A => ⟨adjugate A, by rw [det_adjugate, A.prop, one_pow]⟩⟩
                            /-
                              🎉 no goals
                            -/


instance hasMul : Mul (SpecialLinearGroup n R) :=
                         /-
                           n : Type u
                           inst✝² : DecidableEq n
                           inst✝¹ : Fintype n
                           R : Type v
                           inst✝ : CommRing R
                           A B : Matrix.SpecialLinearGroup n R
                           ⊢ Eq (HMul.hMul ↑A ↑B).det 1
                         -/
  ⟨fun A B => ⟨A * B, by rw [det_mul, A.prop, B.prop, one_mul]⟩⟩
                         /-
                           🎉 no goals
                         -/


instance hasOne : One (SpecialLinearGroup n R) :=
  ⟨⟨1, det_one⟩⟩


instance : Pow (SpecialLinearGroup n R) ℕ where
  pow x n := ⟨x ^ n, (det_pow _ _).trans <| x.prop.symm ▸ one_pow _⟩


instance : Inhabited (SpecialLinearGroup n R) :=
  ⟨1⟩


instance [Fintype R] [DecidableEq R] : Fintype (SpecialLinearGroup n R) := Subtype.fintype _

instance [Finite R] : Finite (SpecialLinearGroup n R) := Subtype.finite


/-- The transpose of a matrix in `SL(n, R)` -/
def transpose (A : SpecialLinearGroup n R) : SpecialLinearGroup n R :=
  ⟨A.1.transpose, A.1.det_transpose ▸ A.2⟩


@[inherit_doc]
scoped postfix:1024 "ᵀ" => SpecialLinearGroup.transpose


theorem coe_mk (A : Matrix n n R) (h : det A = 1) : ↑(⟨A, h⟩ : SpecialLinearGroup n R) = A :=
  rfl


@[simp]
theorem coe_inv : ↑ₘ(A⁻¹) = adjugate A :=
  rfl


@[simp]
theorem coe_mul : ↑ₘ(A * B) = ↑ₘA * ↑ₘB :=
  rfl


@[simp]
theorem coe_one : (1 : SpecialLinearGroup n R) = (1 : Matrix n n R) :=
  rfl


@[simp]
theorem det_coe : det ↑ₘA = 1 :=
  A.2


@[simp]
theorem coe_pow (m : ℕ) : ↑ₘ(A ^ m) = ↑ₘA ^ m :=
  rfl


@[simp]
lemma coe_transpose (A : SpecialLinearGroup n R) : ↑ₘAᵀ = (↑ₘA)ᵀ :=
  rfl


theorem det_ne_zero [Nontrivial R] (g : SpecialLinearGroup n R) : det ↑ₘg ≠ 0 := by
  /-
    n : Type u
    inst✝³ : DecidableEq n
    inst✝² : Fintype n
    R : Type v
    inst✝¹ : CommRing R
    inst✝ : Nontrivial R
    g : Matrix.SpecialLinearGroup n R
    ⊢ Ne (↑g).det 0
  -/
  rw [g.det_coe]
  /-
    n : Type u
    inst✝³ : DecidableEq n
    inst✝² : Fintype n
    R : Type v
    inst✝¹ : CommRing R
    inst✝ : Nontrivial R
    g : Matrix.SpecialLinearGroup n R
    ⊢ Ne 1 0
  -/
  norm_num
  /-
    🎉 no goals
  -/


theorem row_ne_zero [Nontrivial R] (g : SpecialLinearGroup n R) (i : n) : g i ≠ 0 := fun h =>
                                                      /-
                                                        n : Type u
                                                        inst✝³ : DecidableEq n
                                                        inst✝² : Fintype n
                                                        R : Type v
                                                        inst✝¹ : CommRing R
                                                        inst✝ : Nontrivial R
                                                        g : Matrix.SpecialLinearGroup n R
                                                        i : n
                                                        h : Eq (↑g i) 0
                                                        ⊢ ∀ (j : n), Eq (↑g i j) 0
                                                      -/
  g.det_ne_zero <| det_eq_zero_of_row_eq_zero i <| by simp [h]
                                                      /-
                                                        🎉 no goals
                                                      -/


instance monoid : Monoid (SpecialLinearGroup n R) :=
  Function.Injective.monoid _ Subtype.coe_injective coe_one coe_mul coe_pow


instance : Group (SpecialLinearGroup n R) :=
  { SpecialLinearGroup.monoid, SpecialLinearGroup.hasInv with
    inv_mul_cancel := fun A => by
      /-
        n : Type u
        inst✝² : DecidableEq n
        inst✝¹ : Fintype n
        R : Type v
        inst✝ : CommRing R
        A : Matrix.SpecialLinearGroup n R
        ⊢ Eq (HMul.hMul (Inv.inv A) A) 1
      -/
      ext1
      /-
        case a
        n : Type u
        inst✝² : DecidableEq n
        inst✝¹ : Fintype n
        R : Type v
        inst✝ : CommRing R
        A : Matrix.SpecialLinearGroup n R
        i✝ j✝ : n
        ⊢ Eq (↑(HMul.hMul (Inv.inv A) A) i✝ j✝) (↑1 i✝ j✝)
      -/
      simp [adjugate_mul] }
      /-
        🎉 no goals
      -/


/-- A version of `Matrix.toLin' A` that produces linear equivalences. -/
def toLin' : SpecialLinearGroup n R →* (n → R) ≃ₗ[R] n → R where
  toFun A :=
    LinearEquiv.ofLinear (Matrix.toLin' ↑ₘA) (Matrix.toLin' ↑ₘA⁻¹)
          /-
            n : Type u
            inst✝² : DecidableEq n
            inst✝¹ : Fintype n
            R : Type v
            inst✝ : CommRing R
            A : Matrix.SpecialLinearGroup n R
            ⊢ Eq ((Matrix.toLin' ↑A).comp (Matrix.toLin' ↑(Inv.inv A))) LinearMap.id
          -/
      (by rw [← toLin'_mul, ← coe_mul, mul_inv_cancel, coe_one, toLin'_one])
          /-
            🎉 no goals
          -/
          /-
            n : Type u
            inst✝² : DecidableEq n
            inst✝¹ : Fintype n
            R : Type v
            inst✝ : CommRing R
            A : Matrix.SpecialLinearGroup n R
            ⊢ Eq ((Matrix.toLin' ↑(Inv.inv A)).comp (Matrix.toLin' ↑A)) LinearMap.id
          -/
      (by rw [← toLin'_mul, ← coe_mul, inv_mul_cancel, coe_one, toLin'_one])
          /-
            🎉 no goals
          -/
  map_one' := LinearEquiv.toLinearMap_injective Matrix.toLin'_one
  map_mul' A B := LinearEquiv.toLinearMap_injective <| Matrix.toLin'_mul ↑ₘA ↑ₘB


theorem toLin'_apply (A : SpecialLinearGroup n R) (v : n → R) :
    SpecialLinearGroup.toLin' A v = Matrix.toLin' (↑ₘA) v :=
  rfl


theorem toLin'_to_linearMap (A : SpecialLinearGroup n R) :
    ↑(SpecialLinearGroup.toLin' A) = Matrix.toLin' ↑ₘA :=
  rfl


theorem toLin'_symm_apply (A : SpecialLinearGroup n R) (v : n → R) :
    A.toLin'.symm v = Matrix.toLin' (↑ₘA⁻¹) v :=
  rfl


theorem toLin'_symm_to_linearMap (A : SpecialLinearGroup n R) :
    ↑A.toLin'.symm = Matrix.toLin' ↑ₘA⁻¹ :=
  rfl


theorem toLin'_injective :
    Function.Injective ↑(toLin' : SpecialLinearGroup n R →* (n → R) ≃ₗ[R] n → R) := fun _ _ h =>
  Subtype.coe_injective <| Matrix.toLin'.injective <| LinearEquiv.toLinearMap_injective.eq_iff.mpr h


/-- A ring homomorphism from `R` to `S` induces a group homomorphism from
`SpecialLinearGroup n R` to `SpecialLinearGroup n S`. -/
@[simps]
def map (f : R →+* S) : SpecialLinearGroup n R →* SpecialLinearGroup n S where
  toFun g :=
    ⟨f.mapMatrix ↑ₘg, by
      /-
        n : Type u
        inst✝³ : DecidableEq n
        inst✝² : Fintype n
        R : Type v
        inst✝¹ : CommRing R
        S : Type u_1
        inst✝ : CommRing S
        f : RingHom R S
        g : Matrix.SpecialLinearGroup n R
        ⊢ Eq (f.mapMatrix ↑g).det 1
      -/
      rw [← f.map_det]
      /-
        n : Type u
        inst✝³ : DecidableEq n
        inst✝² : Fintype n
        R : Type v
        inst✝¹ : CommRing R
        S : Type u_1
        inst✝ : CommRing S
        f : RingHom R S
        g : Matrix.SpecialLinearGroup n R
        ⊢ Eq (f (↑g).det) 1
      -/
      simp [g.prop]⟩
      /-
        🎉 no goals
      -/
  map_one' := Subtype.ext <| f.mapMatrix.map_one
  map_mul' x y := Subtype.ext <| f.mapMatrix.map_mul ↑ₘx ↑ₘy


@[simp]
theorem center_eq_bot_of_subsingleton [Subsingleton n] :
    center (SpecialLinearGroup n R) = ⊥ :=
                              /-
                                n : Type u
                                inst✝³ : DecidableEq n
                                inst✝² : Fintype n
                                R : Type v
                                inst✝¹ : CommRing R
                                inst✝ : Subsingleton n
                                x : Matrix.SpecialLinearGroup n R
                                x✝ : Membership.mem (Subgroup.center (Matrix.SpecialLinearGroup n R)) x
                                ⊢ Membership.mem Bot.bot x
                              -/
  eq_bot_iff.mpr fun x _ ↦ by rw [mem_bot, Subsingleton.elim x 1]
                              /-
                                🎉 no goals
                              -/


theorem scalar_eq_self_of_mem_center
    {A : SpecialLinearGroup n R} (hA : A ∈ center (SpecialLinearGroup n R)) (i : n) :
    scalar n (A i i) = A := by
  obtain ⟨r : R, hr : scalar n r = A⟩ := mem_range_scalar_of_commute_transvectionStruct fun t ↦
    Subtype.ext_iff.mp <| Subgroup.mem_center_iff.mp hA ⟨t.toMatrix, by simp⟩
  /-
    case intro
    n : Type u
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    R : Type v
    inst✝ : CommRing R
    A : Matrix.SpecialLinearGroup n R
    hA : Membership.mem (Subgroup.center (Matrix.SpecialLinearGroup n R)) A
    i : n
    r : R
    hr : Eq ((Matrix.scalar n) r) ↑A
    ⊢ Eq ((Matrix.scalar n) (↑A i i)) ↑A
  -/
  simp [← congr_fun₂ hr i i, ← hr]
  /-
    🎉 no goals
  -/


theorem scalar_eq_coe_self_center
    (A : center (SpecialLinearGroup n R)) (i : n) :
    scalar n ((A : Matrix n n R) i i) = A :=
  scalar_eq_self_of_mem_center A.property i


/-- The center of a special linear group of degree `n` is the subgroup of scalar matrices, for which
the scalars are the `n`-th roots of unity. -/
theorem mem_center_iff {A : SpecialLinearGroup n R} :
    A ∈ center (SpecialLinearGroup n R) ↔ ∃ (r : R), r ^ (Fintype.card n) = 1 ∧ scalar n r = A := by
  /-
    n : Type u
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    R : Type v
    inst✝ : CommRing R
    A : Matrix.SpecialLinearGroup n R
    ⊢ Iff (Membership.mem (Subgroup.center (Matrix.SpecialLinearGroup n R)) A) (Ex …
  -/
  rcases isEmpty_or_nonempty n with hn | ⟨⟨i⟩⟩; · exact ⟨by aesop, by simp [Subsingleton.elim A 1]⟩
                                                  /-
                                                    🎉 no goals
                                                  -/
  /-
    case inr.intro
    n : Type u
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    R : Type v
    inst✝ : CommRing R
    A : Matrix.SpecialLinearGroup n R
    i : n
    ⊢ Iff (Membership.mem (Subgroup.center (Matrix.SpecialLinearGroup n R)) A) (Ex …
  -/
  refine ⟨fun h ↦ ⟨A i i, ?_, ?_⟩, fun ⟨r, _, hr⟩ ↦ Subgroup.mem_center_iff.mpr fun B ↦ ?_⟩
    /-
      case inr.intro.refine_1
      n : Type u
      inst✝² : DecidableEq n
      inst✝¹ : Fintype n
      R : Type v
      inst✝ : CommRing R
      A : Matrix.SpecialLinearGroup n R
      i : n
      h : Membership.mem (Subgroup.center (Matrix.SpecialLinearGroup n R)) A
      ⊢ Eq (HPow.hPow (↑A i i) (Fintype.card n)) 1
    -/
  · have : det ((scalar n) (A i i)) = 1 := (scalar_eq_self_of_mem_center h i).symm ▸ A.property
    /-
      case inr.intro.refine_1
      n : Type u
      inst✝² : DecidableEq n
      inst✝¹ : Fintype n
      R : Type v
      inst✝ : CommRing R
      A : Matrix.SpecialLinearGroup n R
      i : n
      h : Membership.mem (Subgroup.center (Matrix.SpecialLinearGroup n R)) A
      this : Eq ((Matrix.scalar n) (↑A i i)).det 1
      ⊢ Eq (HPow.hPow (↑A i i) (Fintype.card n)) 1
    -/
    simpa using this
    /-
      🎉 no goals
    -/
    /-
      case inr.intro.refine_2
      n : Type u
      inst✝² : DecidableEq n
      inst✝¹ : Fintype n
      R : Type v
      inst✝ : CommRing R
      A : Matrix.SpecialLinearGroup n R
      i : n
      h : Membership.mem (Subgroup.center (Matrix.SpecialLinearGroup n R)) A
      ⊢ Eq ((Matrix.scalar n) (↑A i i)) ↑A
    -/
  · exact scalar_eq_self_of_mem_center h i
    /-
      🎉 no goals
    -/
    /-
      case inr.intro.refine_3
      n : Type u
      inst✝² : DecidableEq n
      inst✝¹ : Fintype n
      R : Type v
      inst✝ : CommRing R
      A : Matrix.SpecialLinearGroup n R
      i : n
      x✝ : Exists fun r => And (Eq (HPow.hPow r (Fintype.card n)) 1) (Eq ((Matrix.sc …
      r : R
      left✝ : Eq (HPow.hPow r (Fintype.card n)) 1
      hr : Eq ((Matrix.scalar n) r) ↑A
      B : Matrix.SpecialLinearGroup n R
      ⊢ Eq (HMul.hMul B A) (HMul.hMul A B)
    -/
  · suffices ↑ₘ(B * A) = ↑ₘ(A * B) from Subtype.val_injective this
    /-
      case inr.intro.refine_3
      n : Type u
      inst✝² : DecidableEq n
      inst✝¹ : Fintype n
      R : Type v
      inst✝ : CommRing R
      A : Matrix.SpecialLinearGroup n R
      i : n
      x✝ : Exists fun r => And (Eq (HPow.hPow r (Fintype.card n)) 1) (Eq ((Matrix.sc …
      r : R
      left✝ : Eq (HPow.hPow r (Fintype.card n)) 1
      hr : Eq ((Matrix.scalar n) r) ↑A
      B : Matrix.SpecialLinearGroup n R
      ⊢ Eq ↑(HMul.hMul B A) ↑(HMul.hMul A B)
    -/
    simpa only [coe_mul, ← hr] using (scalar_commute (n := n) r (Commute.all r) B).symm
    /-
      🎉 no goals
    -/


/-- An equivalence of groups, from the center of the special linear group to the roots of unity. -/
-- replaced `(Fintype.card n).mkPNat'` by `Fintype.card n` (note `n` is nonempty here)
@[simps]
def center_equiv_rootsOfUnity' (i : n) :
    center (SpecialLinearGroup n R) ≃* rootsOfUnity (Fintype.card n) R where
  toFun A :=
    haveI : Nonempty n := ⟨i⟩
    rootsOfUnity.mkOfPowEq (↑ₘA i i) <| by
      /-
        n : Type u
        inst✝³ : DecidableEq n
        inst✝² : Fintype n
        R : Type v
        inst✝¹ : CommRing R
        S : Type u_1
        inst✝ : CommRing S
        i : n
        A : Subtype fun x => Membership.mem (Subgroup.center (Matrix.SpecialLinearGrou …
        this : Nonempty n
        ⊢ Eq (HPow.hPow (↑↑A i i) (Fintype.card n)) 1
      -/
      obtain ⟨r, hr, hr'⟩ := mem_center_iff.mp A.property
      /-
        case intro.intro
        n : Type u
        inst✝³ : DecidableEq n
        inst✝² : Fintype n
        R : Type v
        inst✝¹ : CommRing R
        S : Type u_1
        inst✝ : CommRing S
        i : n
        A : Subtype fun x => Membership.mem (Subgroup.center (Matrix.SpecialLinearGrou …
        this : Nonempty n
        r : R
        hr : Eq (HPow.hPow r (Fintype.card n)) 1
        hr' : Eq ((Matrix.scalar n) r) ↑↑A
        ⊢ Eq (HPow.hPow (↑↑A i i) (Fintype.card n)) 1
      -/
      replace hr' : A.val i i = r := by simp only [← hr', scalar_apply, diagonal_apply_eq]
      /-
        case intro.intro
        n : Type u
        inst✝³ : DecidableEq n
        inst✝² : Fintype n
        R : Type v
        inst✝¹ : CommRing R
        S : Type u_1
        inst✝ : CommRing S
        i : n
        A : Subtype fun x => Membership.mem (Subgroup.center (Matrix.SpecialLinearGrou …
        this : Nonempty n
        r : R
        hr : Eq (HPow.hPow r (Fintype.card n)) 1
        hr' : Eq (↑↑A i i) r
        ⊢ Eq (HPow.hPow (↑↑A i i) (Fintype.card n)) 1
      -/
      simp only [hr', hr]
      /-
        🎉 no goals
      -/
                                           /-
                                             n : Type u
                                             inst✝³ : DecidableEq n
                                             inst✝² : Fintype n
                                             R : Type v
                                             inst✝¹ : CommRing R
                                             S : Type u_1
                                             inst✝ : CommRing S
                                             i : n
                                             a : Subtype fun x => Membership.mem (rootsOfUnity (Fintype.card n) R) x
                                             ⊢ Eq (HSMul.hSMul a 1).det 1
                                           -/
  invFun a := ⟨⟨a • (1 : Matrix n n R), by aesop⟩,
                                           /-
                                             🎉 no goals
                                           -/
                                                                    /-
                                                                      n : Type u
                                                                      inst✝³ : DecidableEq n
                                                                      inst✝² : Fintype n
                                                                      R : Type v
                                                                      inst✝¹ : CommRing R
                                                                      S : Type u_1
                                                                      inst✝ : CommRing S
                                                                      i : n
                                                                      a : Subtype fun x => Membership.mem (rootsOfUnity (Fintype.card n) R) x
                                                                      B : Matrix.SpecialLinearGroup n R
                                                                      ⊢ Eq ↑(HMul.hMul B ⟨HSMul.hSMul a 1, ⋯⟩) ↑(HMul.hMul ⟨HSMul.hSMul a 1, ⋯⟩ B)
                                                                    -/
    Subgroup.mem_center_iff.mpr fun B ↦ Subtype.val_injective <| by simp [coe_mul]⟩
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
  left_inv A := by
    /-
      n : Type u
      inst✝³ : DecidableEq n
      inst✝² : Fintype n
      R : Type v
      inst✝¹ : CommRing R
      S : Type u_1
      inst✝ : CommRing S
      i : n
      A : Subtype fun x => Membership.mem (Subgroup.center (Matrix.SpecialLinearGrou …
      ⊢ Eq ((fun a => ⟨⟨HSMul.hSMul a 1, ⋯⟩, ⋯⟩) ((fun A => rootsOfUnity.mkOfPowEq ( …
    -/
    refine SetCoe.ext <| SetCoe.ext ?_
    /-
      n : Type u
      inst✝³ : DecidableEq n
      inst✝² : Fintype n
      R : Type v
      inst✝¹ : CommRing R
      S : Type u_1
      inst✝ : CommRing S
      i : n
      A : Subtype fun x => Membership.mem (Subgroup.center (Matrix.SpecialLinearGrou …
      ⊢ Eq ↑↑((fun a => ⟨⟨HSMul.hSMul a 1, ⋯⟩, ⋯⟩) ((fun A => rootsOfUnity.mkOfPowEq …
    -/
    obtain ⟨r, _, hr⟩ := mem_center_iff.mp A.property
    /-
      case intro.intro
      n : Type u
      inst✝³ : DecidableEq n
      inst✝² : Fintype n
      R : Type v
      inst✝¹ : CommRing R
      S : Type u_1
      inst✝ : CommRing S
      i : n
      A : Subtype fun x => Membership.mem (Subgroup.center (Matrix.SpecialLinearGrou …
      r : R
      left✝ : Eq (HPow.hPow r (Fintype.card n)) 1
      hr : Eq ((Matrix.scalar n) r) ↑↑A
      ⊢ Eq ↑↑((fun a => ⟨⟨HSMul.hSMul a 1, ⋯⟩, ⋯⟩) ((fun A => rootsOfUnity.mkOfPowEq …
    -/
    simpa [← hr, Submonoid.smul_def, Units.smul_def] using smul_one_eq_diagonal r
    /-
      🎉 no goals
    -/
  right_inv a := by
    /-
      n : Type u
      inst✝³ : DecidableEq n
      inst✝² : Fintype n
      R : Type v
      inst✝¹ : CommRing R
      S : Type u_1
      inst✝ : CommRing S
      i : n
      a : Subtype fun x => Membership.mem (rootsOfUnity (Fintype.card n) R) x
      ⊢ Eq ((fun A => rootsOfUnity.mkOfPowEq (↑↑A i i) ⋯) ((fun a => ⟨⟨HSMul.hSMul a …
    -/
    obtain ⟨⟨a, _⟩, ha⟩ := a
    /-
      case mk.mk
      n : Type u
      inst✝³ : DecidableEq n
      inst✝² : Fintype n
      R : Type v
      inst✝¹ : CommRing R
      S : Type u_1
      inst✝ : CommRing S
      i : n
      a inv✝ : R
      val_inv✝ : Eq (HMul.hMul a inv✝) 1
      inv_val✝ : Eq (HMul.hMul inv✝ a) 1
      ha : Membership.mem (rootsOfUnity (Fintype.card n) R) { val := a, inv := inv✝, …
      ⊢ Eq ((fun A => rootsOfUnity.mkOfPowEq (↑↑A i i) ⋯) ((fun a => ⟨⟨HSMul.hSMul a …
    -/
    exact SetCoe.ext <| Units.eq_iff.mp <| by simp
    /-
      🎉 no goals
    -/
  map_mul' A B := by
    /-
      n : Type u
      inst✝³ : DecidableEq n
      inst✝² : Fintype n
      R : Type v
      inst✝¹ : CommRing R
      S : Type u_1
      inst✝ : CommRing S
      i : n
      A B : Subtype fun x => Membership.mem (Subgroup.center (Matrix.SpecialLinearGr …
      ⊢ Eq ({ toFun := fun A => rootsOfUnity.mkOfPowEq (↑↑A i i) ⋯, invFun := fun a  …
    -/
    dsimp
    /-
      n : Type u
      inst✝³ : DecidableEq n
      inst✝² : Fintype n
      R : Type v
      inst✝¹ : CommRing R
      S : Type u_1
      inst✝ : CommRing S
      i : n
      A B : Subtype fun x => Membership.mem (Subgroup.center (Matrix.SpecialLinearGr …
      ⊢ Eq (rootsOfUnity.mkOfPowEq (HMul.hMul (↑↑A) (↑↑B) i i) ⋯) (HMul.hMul (rootsO …
    -/
    ext
    /-
      case a.a
      n : Type u
      inst✝³ : DecidableEq n
      inst✝² : Fintype n
      R : Type v
      inst✝¹ : CommRing R
      S : Type u_1
      inst✝ : CommRing S
      i : n
      A B : Subtype fun x => Membership.mem (Subgroup.center (Matrix.SpecialLinearGr …
      ⊢ Eq ↑↑(rootsOfUnity.mkOfPowEq (HMul.hMul (↑↑A) (↑↑B) i i) ⋯) ↑↑(HMul.hMul (ro …
    -/
    simp only [rootsOfUnity.val_mkOfPowEq_coe, Subgroup.coe_mul, Units.val_mul]
    /-
      case a.a
      n : Type u
      inst✝³ : DecidableEq n
      inst✝² : Fintype n
      R : Type v
      inst✝¹ : CommRing R
      S : Type u_1
      inst✝ : CommRing S
      i : n
      A B : Subtype fun x => Membership.mem (Subgroup.center (Matrix.SpecialLinearGr …
      ⊢ Eq (HMul.hMul (↑↑A) (↑↑B) i i) (HMul.hMul (↑↑A i i) (↑↑B i i))
    -/
    rw [← scalar_eq_coe_self_center A i, ← scalar_eq_coe_self_center B i]
    /-
      case a.a
      n : Type u
      inst✝³ : DecidableEq n
      inst✝² : Fintype n
      R : Type v
      inst✝¹ : CommRing R
      S : Type u_1
      inst✝ : CommRing S
      i : n
      A B : Subtype fun x => Membership.mem (Subgroup.center (Matrix.SpecialLinearGr …
      ⊢ Eq (HMul.hMul ((Matrix.scalar n) (↑↑A i i)) ((Matrix.scalar n) (↑↑B i i)) i  …
    -/
    simp
    /-
      🎉 no goals
    -/


open scoped Classical in
/-- An equivalence of groups, from the center of the special linear group to the roots of unity.

See also `center_equiv_rootsOfUnity'`. -/
-- replaced `(Fintype.card n).mkPNat'` by what it means, avoiding `PNat`s.
noncomputable def center_equiv_rootsOfUnity :
    center (SpecialLinearGroup n R) ≃* rootsOfUnity (max (Fintype.card n) 1) R :=
  (isEmpty_or_nonempty n).by_cases
  (fun hn ↦ by
    rw [center_eq_bot_of_subsingleton, Fintype.card_eq_zero, max_eq_right_of_lt zero_lt_one,
      rootsOfUnity_one]
    /-
      n : Type u
      inst✝³ : DecidableEq n
      inst✝² : Fintype n
      R : Type v
      inst✝¹ : CommRing R
      S : Type u_1
      inst✝ : CommRing S
      hn : IsEmpty n
      ⊢ MulEquiv (Subtype fun x => Membership.mem Bot.bot x) (Subtype fun x => Membe …
    -/
    exact MulEquiv.ofUnique)
    /-
      🎉 no goals
    -/
  (fun _ ↦
    (max_eq_left (NeZero.one_le : 1 ≤ Fintype.card n)).symm ▸
      center_equiv_rootsOfUnity' (Classical.arbitrary n))


/-- Coercion of SL `n` `ℤ` to SL `n` `R` for a commutative ring `R`. -/
instance : Coe (SpecialLinearGroup n ℤ) (SpecialLinearGroup n R) :=
  ⟨fun x => map (Int.castRingHom R) x⟩


@[simp]
theorem coe_matrix_coe (g : SpecialLinearGroup n ℤ) :
    ↑(g : SpecialLinearGroup n R) = (↑g : Matrix n n ℤ).map (Int.castRingHom R) :=
  map_apply_coe (Int.castRingHom R) g


/-- Formal operation of negation on special linear group on even cardinality `n` given by negating
each element. -/
instance instNeg : Neg (SpecialLinearGroup n R) :=
  ⟨fun g => ⟨-g, by
    /-
      n : Type u
      inst✝⁴ : DecidableEq n
      inst✝³ : Fintype n
      R : Type v
      inst✝² : CommRing R
      S : Type u_1
      inst✝¹ : CommRing S
      inst✝ : Fact (Even (Fintype.card n))
      g : Matrix.SpecialLinearGroup n R
      ⊢ Eq (Neg.neg ↑g).det 1
    -/
    simpa [(@Fact.out <| Even <| Fintype.card n).neg_one_pow, g.det_coe] using det_smul (↑ₘg) (-1)⟩⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem coe_neg (g : SpecialLinearGroup n R) : ↑(-g) = -(g : Matrix n n R) :=
  rfl


instance : HasDistribNeg (SpecialLinearGroup n R) :=
  Function.Injective.hasDistribNeg _ Subtype.coe_injective coe_neg coe_mul


@[simp]
theorem coe_int_neg (g : SpecialLinearGroup n ℤ) : ↑(-g) = (-↑g : SpecialLinearGroup n R) :=
  Subtype.ext <| (@RingHom.mapMatrix n _ _ _ _ _ _ (Int.castRingHom R)).map_neg ↑g


theorem SL2_inv_expl_det (A : SL(2, R)) :
    det ![![A.1 1 1, -A.1 0 1], ![-A.1 1 0, A.1 0 0]] = 1 := by
  /-
    R : Type v
    inst✝ : CommRing R
    A : Matrix.SpecialLinearGroup (Fin 2) R
    ⊢ Eq (Matrix.det (Matrix.vecCons (Matrix.vecCons (↑A 1 1) (Matrix.vecCons (Neg …
  -/
  rw [Matrix.det_fin_two, mul_comm]
  /-
    R : Type v
    inst✝ : CommRing R
    A : Matrix.SpecialLinearGroup (Fin 2) R
    ⊢ Eq (HSub.hSub (HMul.hMul (Matrix.vecCons (Matrix.vecCons (↑A 1 1) (Matrix.ve …
  -/
  simp only [cons_val_zero, cons_val_one, head_cons, mul_neg, neg_mul, neg_neg]
  /-
    R : Type v
    inst✝ : CommRing R
    A : Matrix.SpecialLinearGroup (Fin 2) R
    ⊢ Eq (HSub.hSub (HMul.hMul (↑A 0 0) (↑A 1 1)) (HMul.hMul (↑A 0 1) (↑A 1 0))) 1
  -/
  have := A.2
  /-
    R : Type v
    inst✝ : CommRing R
    A : Matrix.SpecialLinearGroup (Fin 2) R
    this : Eq (↑A).det 1
    ⊢ Eq (HSub.hSub (HMul.hMul (↑A 0 0) (↑A 1 1)) (HMul.hMul (↑A 0 1) (↑A 1 0))) 1
  -/
  rw [Matrix.det_fin_two] at this
  /-
    R : Type v
    inst✝ : CommRing R
    A : Matrix.SpecialLinearGroup (Fin 2) R
    this : Eq (HSub.hSub (HMul.hMul (↑A 0 0) (↑A 1 1)) (HMul.hMul (↑A 0 1) (↑A 1 0 …
    ⊢ Eq (HSub.hSub (HMul.hMul (↑A 0 0) (↑A 1 1)) (HMul.hMul (↑A 0 1) (↑A 1 0))) 1
  -/
  convert this
  /-
    🎉 no goals
  -/


theorem SL2_inv_expl (A : SL(2, R)) :
    A⁻¹ = ⟨![![A.1 1 1, -A.1 0 1], ![-A.1 1 0, A.1 0 0]], SL2_inv_expl_det A⟩ := by
  /-
    R : Type v
    inst✝ : CommRing R
    A : Matrix.SpecialLinearGroup (Fin 2) R
    ⊢ Eq (Inv.inv A) ⟨Matrix.vecCons (Matrix.vecCons (↑A 1 1) (Matrix.vecCons (Neg …
  -/
  ext
  /-
    case a
    R : Type v
    inst✝ : CommRing R
    A : Matrix.SpecialLinearGroup (Fin 2) R
    i✝ j✝ : Fin 2
    ⊢ Eq (↑(Inv.inv A) i✝ j✝) (↑⟨Matrix.vecCons (Matrix.vecCons (↑A 1 1) (Matrix.v …
  -/
  have := Matrix.adjugate_fin_two A.1
  /-
    case a
    R : Type v
    inst✝ : CommRing R
    A : Matrix.SpecialLinearGroup (Fin 2) R
    i✝ j✝ : Fin 2
    this : Eq (↑A).adjugate (Matrix.of (Matrix.vecCons (Matrix.vecCons (↑A 1 1) (M …
    ⊢ Eq (↑(Inv.inv A) i✝ j✝) (↑⟨Matrix.vecCons (Matrix.vecCons (↑A 1 1) (Matrix.v …
  -/
  rw [coe_inv, this]
  /-
    case a
    R : Type v
    inst✝ : CommRing R
    A : Matrix.SpecialLinearGroup (Fin 2) R
    i✝ j✝ : Fin 2
    this : Eq (↑A).adjugate (Matrix.of (Matrix.vecCons (Matrix.vecCons (↑A 1 1) (M …
    ⊢ Eq (Matrix.of (Matrix.vecCons (Matrix.vecCons (↑A 1 1) (Matrix.vecCons (Neg. …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem fin_two_induction (P : SL(2, R) → Prop)
                                                                           /-
                                                                             n : Type u
                                                                             inst✝³ : DecidableEq n
                                                                             inst✝² : Fintype n
                                                                             R : Type v
                                                                             inst✝¹ : CommRing R
                                                                             S : Type u_1
                                                                             inst✝ : CommRing S
                                                                             P : Matrix.SpecialLinearGroup (Fin 2) R → Prop
                                                                             a b c d : R
                                                                             hdet : Eq (HSub.hSub (HMul.hMul a d) (HMul.hMul b c)) 1
                                                                             ⊢ Eq (Matrix.of (Matrix.vecCons (Matrix.vecCons a (Matrix.vecCons b Matrix.vec …
                                                                           -/
    (h : ∀ (a b c d : R) (hdet : a * d - b * c = 1), P ⟨!![a, b; c, d], by rwa [det_fin_two_of]⟩)
                                                                           /-
                                                                             🎉 no goals
                                                                           -/
    (g : SL(2, R)) : P g := by
  /-
    R : Type v
    inst✝ : CommRing R
    P : Matrix.SpecialLinearGroup (Fin 2) R → Prop
    h : ∀ (a b c d : R) (hdet : Eq (HSub.hSub (HMul.hMul a d) (HMul.hMul b c)) 1), …
    g : Matrix.SpecialLinearGroup (Fin 2) R
    ⊢ P g
  -/
  obtain ⟨m, hm⟩ := g
  /-
    case mk
    R : Type v
    inst✝ : CommRing R
    P : Matrix.SpecialLinearGroup (Fin 2) R → Prop
    h : ∀ (a b c d : R) (hdet : Eq (HSub.hSub (HMul.hMul a d) (HMul.hMul b c)) 1), …
    m : Matrix (Fin 2) (Fin 2) R
    hm : Eq m.det 1
    ⊢ P ⟨m, hm⟩
  -/
  convert h (m 0 0) (m 0 1) (m 1 0) (m 1 1) (by rwa [det_fin_two] at hm)
  /-
    case h.e'_1.h.e'_3
    R : Type v
    inst✝ : CommRing R
    P : Matrix.SpecialLinearGroup (Fin 2) R → Prop
    h : ∀ (a b c d : R) (hdet : Eq (HSub.hSub (HMul.hMul a d) (HMul.hMul b c)) 1), …
    m : Matrix (Fin 2) (Fin 2) R
    hm : Eq m.det 1
    ⊢ Eq m (Matrix.of (Matrix.vecCons (Matrix.vecCons (m 0 0) (Matrix.vecCons (m 0 …
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
  ext i j; fin_cases i <;> fin_cases j <;> rfl
                                           /-
                                             🎉 no goals
                                           -/


theorem fin_two_exists_eq_mk_of_apply_zero_one_eq_zero {R : Type*} [Field R] (g : SL(2, R))
    (hg : g 1 0 = 0) :
                                                        /-
                                                          n : Type u
                                                          inst✝⁴ : DecidableEq n
                                                          inst✝³ : Fintype n
                                                          R✝ : Type v
                                                          inst✝² : CommRing R✝
                                                          S : Type u_1
                                                          inst✝¹ : CommRing S
                                                          R : Type u_2
                                                          inst✝ : Field R
                                                          g : Matrix.SpecialLinearGroup (Fin 2) R
                                                          hg : Eq (↑g 1 0) 0
                                                          a b : R
                                                          h : Ne a 0
                                                          ⊢ Eq (Matrix.of (Matrix.vecCons (Matrix.vecCons a (Matrix.vecCons b Matrix.vec …
                                                        -/
    ∃ (a b : R) (h : a ≠ 0), g = (⟨!![a, b; 0, a⁻¹], by simp [h]⟩ : SL(2, R)) := by
                                                        /-
                                                          🎉 no goals
                                                        -/
  /-
    R : Type u_2
    inst✝ : Field R
    g : Matrix.SpecialLinearGroup (Fin 2) R
    hg : Eq (↑g 1 0) 0
    ⊢ Exists fun a => Exists fun b => Exists fun h => Eq g ⟨Matrix.of (Matrix.vecC …
  -/
  induction' g using Matrix.SpecialLinearGroup.fin_two_induction with a b c d h_det
  /-
    case h
    R : Type u_2
    inst✝ : Field R
    a b c d : R
    h_det : Eq (HSub.hSub (HMul.hMul a d) (HMul.hMul b c)) 1
    hg : Eq (↑⟨Matrix.of (Matrix.vecCons (Matrix.vecCons a (Matrix.vecCons b Matri …
    ⊢ Exists fun a_1 => Exists fun b_1 => Exists fun h => Eq ⟨Matrix.of (Matrix.ve …
  -/
  replace hg : c = 0 := by simpa using hg
  /-
    case h
    R : Type u_2
    inst✝ : Field R
    a b c d : R
    h_det : Eq (HSub.hSub (HMul.hMul a d) (HMul.hMul b c)) 1
    hg : Eq c 0
    ⊢ Exists fun a_1 => Exists fun b_1 => Exists fun h => Eq ⟨Matrix.of (Matrix.ve …
  -/
  have had : a * d = 1 := by rwa [hg, mul_zero, sub_zero] at h_det
  /-
    case h
    R : Type u_2
    inst✝ : Field R
    a b c d : R
    h_det : Eq (HSub.hSub (HMul.hMul a d) (HMul.hMul b c)) 1
    hg : Eq c 0
    had : Eq (HMul.hMul a d) 1
    ⊢ Exists fun a_1 => Exists fun b_1 => Exists fun h => Eq ⟨Matrix.of (Matrix.ve …
  -/
  refine ⟨a, b, left_ne_zero_of_mul_eq_one had, ?_⟩
  /-
    case h
    R : Type u_2
    inst✝ : Field R
    a b c d : R
    h_det : Eq (HSub.hSub (HMul.hMul a d) (HMul.hMul b c)) 1
    hg : Eq c 0
    had : Eq (HMul.hMul a d) 1
    ⊢ Eq ⟨Matrix.of (Matrix.vecCons (Matrix.vecCons a (Matrix.vecCons b Matrix.vec …
  -/
  simp_rw [eq_inv_of_mul_eq_one_right had, hg]
  /-
    🎉 no goals
  -/


lemma isCoprime_row (A : SL(2, R)) (i : Fin 2) : IsCoprime (A i 0) (A i 1) := by
  refine match i with
  | 0 => ⟨A 1 1, -(A 1 0), ?_⟩
  | 1 => ⟨-(A 0 1), A 0 0, ?_⟩ <;>
    /-
      case refine_1
      R : Type v
      inst✝ : CommRing R
      A : Matrix.SpecialLinearGroup (Fin 2) R
      i : Fin 2
      ⊢ Eq (HAdd.hAdd (HMul.hMul (↑A 1 1) (↑A 0 0)) (HMul.hMul (Neg.neg (↑A 1 0)) (↑ …
    -/
    /-
      case refine_1
      R : Type v
      inst✝ : CommRing R
      A : Matrix.SpecialLinearGroup (Fin 2) R
      i : Fin 2
      ⊢ Eq (HAdd.hAdd (HMul.hMul (↑A 1 1) (↑A 0 0)) (HMul.hMul (Neg.neg (↑A 1 0)) (↑ …
    -/
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type v
      inst✝ : CommRing R
      A : Matrix.SpecialLinearGroup (Fin 2) R
      i : Fin 2
      ⊢ Eq (HAdd.hAdd (HMul.hMul (Neg.neg (↑A 0 1)) (↑A 1 0)) (HMul.hMul (↑A 0 0) (↑ …
    -/
    ring
    /-
      🎉 no goals
    -/


lemma isCoprime_col (A : SL(2, R)) (j : Fin 2) : IsCoprime (A 0 j) (A 1 j) := by
  refine match j with
  | 0 => ⟨A 1 1, -(A 0 1), ?_⟩
  | 1 => ⟨-(A 1 0), A 0 0, ?_⟩ <;>
    /-
      case refine_1
      R : Type v
      inst✝ : CommRing R
      A : Matrix.SpecialLinearGroup (Fin 2) R
      j : Fin 2
      ⊢ Eq (HAdd.hAdd (HMul.hMul (↑A 1 1) (↑A 0 0)) (HMul.hMul (Neg.neg (↑A 0 1)) (↑ …
    -/
    /-
      case refine_1
      R : Type v
      inst✝ : CommRing R
      A : Matrix.SpecialLinearGroup (Fin 2) R
      j : Fin 2
      ⊢ Eq (HAdd.hAdd (HMul.hMul (↑A 1 1) (↑A 0 0)) (HMul.hMul (Neg.neg (↑A 0 1)) (↑ …
    -/
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type v
      inst✝ : CommRing R
      A : Matrix.SpecialLinearGroup (Fin 2) R
      j : Fin 2
      ⊢ Eq (HAdd.hAdd (HMul.hMul (Neg.neg (↑A 1 0)) (↑A 0 1)) (HMul.hMul (↑A 0 0) (↑ …
    -/
    ring
    /-
      🎉 no goals
    -/


/-- Given any pair of coprime elements of `R`, there exists a matrix in `SL(2, R)` having those
entries as its left or right column. -/
lemma exists_SL2_col {a b : R} (hab : IsCoprime a b) (j : Fin 2) :
    ∃ g : SL(2, R), g 0 j = a ∧ g 1 j = b := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    a b : R
    hab : IsCoprime a b
    j : Fin 2
    ⊢ Exists fun g => And (Eq (↑g 0 j) a) (Eq (↑g 1 j) b)
  -/
  obtain ⟨u, v, h⟩ := hab
  refine match j with
  | 0 => ⟨⟨!![a, -v; b, u], ?_⟩, rfl, rfl⟩
  | 1 => ⟨⟨!![v, a; -u, b], ?_⟩, rfl, rfl⟩ <;>
    /-
      case intro.intro.refine_1
      R : Type u_1
      inst✝ : CommRing R
      a b : R
      j : Fin 2
      u v : R
      h : Eq (HAdd.hAdd (HMul.hMul u a) (HMul.hMul v b)) 1
      ⊢ Eq (Matrix.of (Matrix.vecCons (Matrix.vecCons a (Matrix.vecCons (Neg.neg v)  …
    -/
    /-
      case intro.intro.refine_1
      R : Type u_1
      inst✝ : CommRing R
      a b : R
      j : Fin 2
      u v : R
      h : Eq (HAdd.hAdd (HMul.hMul u a) (HMul.hMul v b)) 1
      ⊢ Eq (HSub.hSub (HMul.hMul a u) (HMul.hMul (Neg.neg v) b)) (HAdd.hAdd (HMul.hM …
    -/
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.refine_2
      R : Type u_1
      inst✝ : CommRing R
      a b : R
      j : Fin 2
      u v : R
      h : Eq (HAdd.hAdd (HMul.hMul u a) (HMul.hMul v b)) 1
      ⊢ Eq (HSub.hSub (HMul.hMul v b) (HMul.hMul a (Neg.neg u))) (HAdd.hAdd (HMul.hM …
    -/
    ring
    /-
      🎉 no goals
    -/


/-- Given any pair of coprime elements of `R`, there exists a matrix in `SL(2, R)` having those
entries as its top or bottom row. -/
lemma exists_SL2_row {a b : R} (hab : IsCoprime a b) (i : Fin 2) :
    ∃ g : SL(2, R), g i 0 = a ∧ g i 1 = b := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    a b : R
    hab : IsCoprime a b
    i : Fin 2
    ⊢ Exists fun g => And (Eq (↑g i 0) a) (Eq (↑g i 1) b)
  -/
  obtain ⟨u, v, h⟩ := hab
  refine match i with
  | 0 => ⟨⟨!![a, b; -v, u], ?_⟩, rfl, rfl⟩
  | 1 => ⟨⟨!![v, -u; a, b], ?_⟩, rfl, rfl⟩ <;>
    /-
      case intro.intro.refine_1
      R : Type u_1
      inst✝ : CommRing R
      a b : R
      i : Fin 2
      u v : R
      h : Eq (HAdd.hAdd (HMul.hMul u a) (HMul.hMul v b)) 1
      ⊢ Eq (Matrix.of (Matrix.vecCons (Matrix.vecCons a (Matrix.vecCons b Matrix.vec …
    -/
    /-
      case intro.intro.refine_1
      R : Type u_1
      inst✝ : CommRing R
      a b : R
      i : Fin 2
      u v : R
      h : Eq (HAdd.hAdd (HMul.hMul u a) (HMul.hMul v b)) 1
      ⊢ Eq (HSub.hSub (HMul.hMul a u) (HMul.hMul b (Neg.neg v))) (HAdd.hAdd (HMul.hM …
    -/
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.refine_2
      R : Type u_1
      inst✝ : CommRing R
      a b : R
      i : Fin 2
      u v : R
      h : Eq (HAdd.hAdd (HMul.hMul u a) (HMul.hMul v b)) 1
      ⊢ Eq (HSub.hSub (HMul.hMul v b) (HMul.hMul (Neg.neg u) a)) (HAdd.hAdd (HMul.hM …
    -/
    ring
    /-
      🎉 no goals
    -/


/-- A vector with coprime entries, right-multiplied by a matrix in `SL(2, R)`, has
coprime entries. -/
lemma vecMulSL {v : Fin 2 → R} (hab : IsCoprime (v 0) (v 1)) (A : SL(2, R)) :
    IsCoprime ((v ᵥ* A.1) 0) ((v ᵥ* A.1) 1) := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    v : Fin 2 → R
    hab : IsCoprime (v 0) (v 1)
    A : Matrix.SpecialLinearGroup (Fin 2) R
    ⊢ IsCoprime (Matrix.vecMul v (↑A) 0) (Matrix.vecMul v (↑A) 1)
  -/
  obtain ⟨g, hg⟩ := hab.exists_SL2_row 0
  /-
    case intro
    R : Type u_1
    inst✝ : CommRing R
    v : Fin 2 → R
    hab : IsCoprime (v 0) (v 1)
    A g : Matrix.SpecialLinearGroup (Fin 2) R
    hg : And (Eq (↑g 0 0) (v 0)) (Eq (↑g 0 1) (v 1))
    ⊢ IsCoprime (Matrix.vecMul v (↑A) 0) (Matrix.vecMul v (↑A) 1)
  -/
  have : v = g 0 := funext fun t ↦ by { fin_cases t <;> tauto }
  /-
    case intro
    R : Type u_1
    inst✝ : CommRing R
    v : Fin 2 → R
    hab : IsCoprime (v 0) (v 1)
    A g : Matrix.SpecialLinearGroup (Fin 2) R
    hg : And (Eq (↑g 0 0) (v 0)) (Eq (↑g 0 1) (v 1))
    this : Eq v (↑g 0)
    ⊢ IsCoprime (Matrix.vecMul v (↑A) 0) (Matrix.vecMul v (↑A) 1)
  -/
  simpa only [this] using isCoprime_row (g * A) 0
  /-
    🎉 no goals
  -/


/-- A vector with coprime entries, left-multiplied by a matrix in `SL(2, R)`, has
coprime entries. -/
lemma mulVecSL {v : Fin 2 → R} (hab : IsCoprime (v 0) (v 1)) (A : SL(2, R)) :
    IsCoprime ((A.1 *ᵥ v) 0) ((A.1 *ᵥ v) 1) := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    v : Fin 2 → R
    hab : IsCoprime (v 0) (v 1)
    A : Matrix.SpecialLinearGroup (Fin 2) R
    ⊢ IsCoprime ((↑A).mulVec v 0) ((↑A).mulVec v 1)
  -/
  simpa only [← vecMul_transpose] using hab.vecMulSL A.transpose
  /-
    🎉 no goals
  -/


/-- The matrix `S = [[0, -1], [1, 0]]` as an element of `SL(2, ℤ)`.

This element acts naturally on the Euclidean plane as a rotation about the origin by `π / 2`.

This element also acts naturally on the hyperbolic plane as rotation about `i` by `π`. It
represents the Mobiüs transformation `z ↦ -1/z` and is an involutive elliptic isometry. -/
def S : SL(2, ℤ) :=
                       /-
                         ⊢ Eq (Matrix.of (Matrix.vecCons (Matrix.vecCons 0 (Matrix.vecCons (-1) Matrix. …
                       -/
  ⟨!![0, -1; 1, 0], by norm_num [Matrix.det_fin_two_of]⟩
                       /-
                         🎉 no goals
                       -/


/-- The matrix `T = [[1, 1], [0, 1]]` as an element of `SL(2, ℤ)`. -/
def T : SL(2, ℤ) :=
                      /-
                        ⊢ Eq (Matrix.of (Matrix.vecCons (Matrix.vecCons 1 (Matrix.vecCons 1 Matrix.vec …
                      -/
  ⟨!![1, 1; 0, 1], by norm_num [Matrix.det_fin_two_of]⟩
                      /-
                        🎉 no goals
                      -/


theorem coe_S : ↑S = !![0, -1; 1, 0] :=
  rfl


theorem coe_T : ↑T = (!![1, 1; 0, 1] : Matrix _ _ ℤ) :=
  rfl


                                                   /-
                                                     ⊢ Eq (↑(Inv.inv ModularGroup.T)) (Matrix.of (Matrix.vecCons (Matrix.vecCons 1  …
                                                   -/
theorem coe_T_inv : ↑(T⁻¹) = !![1, -1; 0, 1] := by simp [coe_inv, coe_T, adjugate_fin_two]
                                                   /-
                                                     🎉 no goals
                                                   -/


theorem coe_T_zpow (n : ℤ) : (T ^ n).1 = !![1, n; 0, 1] := by
  /-
    n : Int
    ⊢ Eq (↑(HPow.hPow ModularGroup.T n)) (Matrix.of (Matrix.vecCons (Matrix.vecCon …
  -/
  induction' n using Int.induction_on with n h n h
    /-
      case hz
      ⊢ Eq (↑(HPow.hPow ModularGroup.T 0)) (Matrix.of (Matrix.vecCons (Matrix.vecCon …
    -/
  · rw [zpow_zero, coe_one, Matrix.one_fin_two]
    /-
      🎉 no goals
    -/
    /-
      case hp
      n : Nat
      h : Eq (↑(HPow.hPow ModularGroup.T ↑n)) (Matrix.of (Matrix.vecCons (Matrix.vec …
      ⊢ Eq (↑(HPow.hPow ModularGroup.T (HAdd.hAdd (↑n) 1))) (Matrix.of (Matrix.vecCo …
    -/
  · simp_rw [zpow_add, zpow_one, coe_mul, h, coe_T, Matrix.mul_fin_two]
    /-
      case hp
      n : Nat
      h : Eq (↑(HPow.hPow ModularGroup.T ↑n)) (Matrix.of (Matrix.vecCons (Matrix.vec …
      ⊢ Eq (Matrix.of (Matrix.vecCons (Matrix.vecCons (HAdd.hAdd (HMul.hMul 1 1) (HM …
    -/
    congrm !![_, ?_; _, _]
    /-
      case hp
      n : Nat
      h : Eq (↑(HPow.hPow ModularGroup.T ↑n)) (Matrix.of (Matrix.vecCons (Matrix.vec …
      ⊢ Eq (HAdd.hAdd (HMul.hMul 1 1) (HMul.hMul (↑n) 1)) (HAdd.hAdd (↑n) 1)
    -/
    rw [mul_one, mul_one, add_comm]
    /-
      🎉 no goals
    -/
    /-
      case hn
      n : Nat
      h : Eq (↑(HPow.hPow ModularGroup.T (Neg.neg ↑n))) (Matrix.of (Matrix.vecCons ( …
      ⊢ Eq (↑(HPow.hPow ModularGroup.T (HSub.hSub (Neg.neg ↑n) 1))) (Matrix.of (Matr …
    -/
  · simp_rw [zpow_sub, zpow_one, coe_mul, h, coe_T_inv, Matrix.mul_fin_two]
    /-
      case hn
      n : Nat
      h : Eq (↑(HPow.hPow ModularGroup.T (Neg.neg ↑n))) (Matrix.of (Matrix.vecCons ( …
      ⊢ Eq (Matrix.of (Matrix.vecCons (Matrix.vecCons (HAdd.hAdd (HMul.hMul 1 1) (HM …
    -/
                                /-
                                  🎉 no goals
                                -/
    congrm !![?_, ?_; _, _] <;> ring
                                /-
                                  🎉 no goals
                                -/


@[simp]
theorem T_pow_mul_apply_one (n : ℤ) (g : SL(2, ℤ)) : (T ^ n * g) 1 = g 1 := by
  /-
    n : Int
    g : Matrix.SpecialLinearGroup (Fin 2) Int
    ⊢ Eq (↑(HMul.hMul (HPow.hPow ModularGroup.T n) g) 1) (↑g 1)
  -/
  ext j
  /-
    case h
    n : Int
    g : Matrix.SpecialLinearGroup (Fin 2) Int
    j : Fin 2
    ⊢ Eq (↑(HMul.hMul (HPow.hPow ModularGroup.T n) g) 1 j) (↑g 1 j)
  -/
  simp [coe_T_zpow, Matrix.vecMul, dotProduct, Fin.sum_univ_succ, vecTail]
  /-
    🎉 no goals
  -/


@[simp]
theorem T_mul_apply_one (g : SL(2, ℤ)) : (T * g) 1 = g 1 := by
  /-
    g : Matrix.SpecialLinearGroup (Fin 2) Int
    ⊢ Eq (↑(HMul.hMul ModularGroup.T g) 1) (↑g 1)
  -/
  simpa using T_pow_mul_apply_one 1 g
  /-
    🎉 no goals
  -/


@[simp]
theorem T_inv_mul_apply_one (g : SL(2, ℤ)) : (T⁻¹ * g) 1 = g 1 := by
  /-
    g : Matrix.SpecialLinearGroup (Fin 2) Int
    ⊢ Eq (↑(HMul.hMul (Inv.inv ModularGroup.T) g) 1) (↑g 1)
  -/
  simpa using T_pow_mul_apply_one (-1) g
  /-
    🎉 no goals
  -/


lemma S_mul_S_eq : (S : Matrix (Fin 2) (Fin 2) ℤ) * S = -1 := by
  simp only [S, Int.reduceNeg, pow_two, coe_mul, cons_mul, Nat.succ_eq_add_one, Nat.reduceAdd,
    vecMul_cons, head_cons, zero_smul, tail_cons, neg_smul, one_smul, neg_cons, neg_zero, neg_empty,
    empty_vecMul, add_zero, zero_add, empty_mul, Equiv.symm_apply_apply]
  /-
    ⊢ Eq (Matrix.of (Matrix.vecCons (Matrix.vecCons (-1) (Matrix.vecCons 0 Matrix. …
  -/
  exact Eq.symm (eta_fin_two (-1))
  /-
    🎉 no goals
  -/


lemma T_S_rel : S • S • S • T • S • T • S = T⁻¹ := by
  /-
    ⊢ Eq (HSMul.hSMul ModularGroup.S (HSMul.hSMul ModularGroup.S (HSMul.hSMul Modu …
  -/
  ext i j
  /-
    case a
    i j : Fin 2
    ⊢ Eq (↑(HSMul.hSMul ModularGroup.S (HSMul.hSMul ModularGroup.S (HSMul.hSMul Mo …
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
  fin_cases i <;> fin_cases j <;> rfl
                                  /-
                                    🎉 no goals
                                  -/


