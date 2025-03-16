theorem isAlgClosure_of_transcendence_basis [IsAlgClosed K] (hv : IsTranscendenceBasis R v) :
    IsAlgClosure (Algebra.adjoin R (Set.range v)) K :=
  letI := RingHom.domain_nontrivial (algebraMap R K)
                      /-
                        R : Type u_1
                        K : Type u_3
                        inst✝³ : CommRing R
                        inst✝² : Field K
                        inst✝¹ : Algebra R K
                        ι : Type u_4
                        v : ι → K
                        inst✝ : IsAlgClosed K
                        hv : IsTranscendenceBasis R v
                        this : Nontrivial R := RingHom.domain_nontrivial (algebraMap R K)
                        ⊢ IsAlgClosed K
                      -/
  { isAlgClosed := by infer_instance
                      /-
                        🎉 no goals
                      -/
    isAlgebraic := hv.isAlgebraic }


/-- setting `R` to be `ZMod (ringChar R)` this result shows that if two algebraically
closed fields have equipotent transcendence bases and the same characteristic then they are
isomorphic. -/
def equivOfTranscendenceBasis [IsAlgClosed K] [IsAlgClosed L] (e : ι ≃ κ)
    (hv : IsTranscendenceBasis R v) (hw : IsTranscendenceBasis R w) : K ≃+* L := by
  /-
    R : Type u_1
    L : Type u_2
    K : Type u_3
    inst✝⁶ : CommRing R
    inst✝⁵ : Field K
    inst✝⁴ : Algebra R K
    inst✝³ : Field L
    inst✝² : Algebra R L
    ι : Type u_4
    v : ι → K
    κ : Type u_5
    w : κ → L
    hv✝ : AlgebraicIndependent R v
    hw✝ : AlgebraicIndependent R w
    inst✝¹ : IsAlgClosed K
    inst✝ : IsAlgClosed L
    e : Equiv ι κ
    hv : IsTranscendenceBasis R v
    hw : IsTranscendenceBasis R w
    ⊢ RingEquiv K L
  -/
  letI := isAlgClosure_of_transcendence_basis v hv
  /-
    R : Type u_1
    L : Type u_2
    K : Type u_3
    inst✝⁶ : CommRing R
    inst✝⁵ : Field K
    inst✝⁴ : Algebra R K
    inst✝³ : Field L
    inst✝² : Algebra R L
    ι : Type u_4
    v : ι → K
    κ : Type u_5
    w : κ → L
    hv✝ : AlgebraicIndependent R v
    hw✝ : AlgebraicIndependent R w
    inst✝¹ : IsAlgClosed K
    inst✝ : IsAlgClosed L
    e : Equiv ι κ
    hv : IsTranscendenceBasis R v
    hw : IsTranscendenceBasis R w
    this : IsAlgClosure (Subtype fun x => Membership.mem (Algebra.adjoin R (Set.ra …
    ⊢ RingEquiv K L
  -/
  letI := isAlgClosure_of_transcendence_basis w hw
  have e : Algebra.adjoin R (Set.range v) ≃+* Algebra.adjoin R (Set.range w) := by
    refine hv.1.aevalEquiv.symm.toRingEquiv.trans ?_
    refine (AlgEquiv.ofAlgHom (MvPolynomial.rename e)
      (MvPolynomial.rename e.symm) ?_ ?_).toRingEquiv.trans ?_
    · ext; simp
    · ext; simp
    exact hw.1.aevalEquiv.toRingEquiv
  /-
    R : Type u_1
    L : Type u_2
    K : Type u_3
    inst✝⁶ : CommRing R
    inst✝⁵ : Field K
    inst✝⁴ : Algebra R K
    inst✝³ : Field L
    inst✝² : Algebra R L
    ι : Type u_4
    v : ι → K
    κ : Type u_5
    w : κ → L
    hv✝ : AlgebraicIndependent R v
    hw✝ : AlgebraicIndependent R w
    inst✝¹ : IsAlgClosed K
    inst✝ : IsAlgClosed L
    e✝ : Equiv ι κ
    hv : IsTranscendenceBasis R v
    hw : IsTranscendenceBasis R w
    this✝ : IsAlgClosure (Subtype fun x => Membership.mem (Algebra.adjoin R (Set.r …
    this : IsAlgClosure (Subtype fun x => Membership.mem (Algebra.adjoin R (Set.ra …
    e : RingEquiv (Subtype fun x => Membership.mem (Algebra.adjoin R (Set.range v) …
    ⊢ RingEquiv K L
  -/
  exact IsAlgClosure.equivOfEquiv K L e
  /-
    🎉 no goals
  -/


/-- The cardinality of an algebraically closed `R`-algebra is less than or equal to
the maximum of of the cardinality of `R`, the cardinality of a transcendence basis and
`ℵ₀`

For a simpler, but less universe-polymorphic statement, see
`IsAlgClosed.cardinal_le_max_transcendence_basis'`  -/
theorem cardinal_le_max_transcendence_basis (hv : IsTranscendenceBasis R v) :
    Cardinal.lift.{max u w} #K ≤ max (max (Cardinal.lift.{max v w} #R)
      (Cardinal.lift.{max u v} #ι)) ℵ₀ :=
  calc
    Cardinal.lift.{max u w} #K ≤ Cardinal.lift.{max u w}
        (max #(Algebra.adjoin R (Set.range v)) ℵ₀) := by
      /-
        R : Type u
        K : Type v
        inst✝³ : CommRing R
        inst✝² : Field K
        inst✝¹ : Algebra R K
        inst✝ : IsAlgClosed K
        ι : Type w
        v : ι → K
        hv : IsTranscendenceBasis R v
        ⊢ LE.le (Cardinal.lift.{max u w, v} (Cardinal.mk K)) (Cardinal.lift.{max u w,  …
      -/
      letI := isAlgClosure_of_transcendence_basis v hv
      /-
        R : Type u
        K : Type v
        inst✝³ : CommRing R
        inst✝² : Field K
        inst✝¹ : Algebra R K
        inst✝ : IsAlgClosed K
        ι : Type w
        v : ι → K
        hv : IsTranscendenceBasis R v
        this : IsAlgClosure (Subtype fun x => Membership.mem (Algebra.adjoin R (Set.ra …
        ⊢ LE.le (Cardinal.lift.{max u w, v} (Cardinal.mk K)) (Cardinal.lift.{max u w,  …
      -/
      simpa using Algebra.IsAlgebraic.cardinalMk_le_max (Algebra.adjoin R (Set.range v)) K
      /-
        🎉 no goals
      -/
    _ = Cardinal.lift.{v} (max #(MvPolynomial ι R) ℵ₀) := by
      rw [lift_max, ← Cardinal.lift_mk_eq.2 ⟨hv.1.aevalEquiv.toEquiv⟩, lift_aleph0,
        ← lift_aleph0.{max u v w, max u w}, ← lift_max, lift_umax.{max u w, v}]
    _ ≤ Cardinal.lift.{v} (max (max (max (Cardinal.lift #R) (Cardinal.lift #ι)) ℵ₀) ℵ₀) :=
        lift_le.2 (max_le_max MvPolynomial.cardinalMk_le_max_lift le_rfl)
                /-
                  R : Type u
                  K : Type v
                  inst✝³ : CommRing R
                  inst✝² : Field K
                  inst✝¹ : Algebra R K
                  inst✝ : IsAlgClosed K
                  ι : Type w
                  v : ι → K
                  hv : IsTranscendenceBasis R v
                  ⊢ Eq (Cardinal.lift.{v, max u w} (Max.max (Max.max (Max.max (Cardinal.lift.{w, …
                -/
    _ = _ := by simp
                /-
                  🎉 no goals
                -/


/-- The cardinality of an algebraically closed `R`-algebra is less than or equal to
the maximum of of the cardinality of `R`, the cardinality of a transcendence basis and
`ℵ₀`

A less-universe polymorphic, but simpler statement of
`IsAlgClosed.cardinal_le_max_transcendence_basis`  -/
theorem cardinal_le_max_transcendence_basis' (hv : IsTranscendenceBasis R v') :
    #K' ≤ max (max #R #ι') ℵ₀ := by
  /-
    R : Type u
    inst✝³ : CommRing R
    K' : Type u
    inst✝² : Field K'
    inst✝¹ : Algebra R K'
    inst✝ : IsAlgClosed K'
    ι' : Type u
    v' : ι' → K'
    hv : IsTranscendenceBasis R v'
    ⊢ LE.le (Cardinal.mk K') (Max.max (Max.max (Cardinal.mk R) (Cardinal.mk ι')) C …
  -/
  simpa using cardinal_le_max_transcendence_basis v' hv
  /-
    🎉 no goals
  -/


/-- If `K` is an uncountable algebraically closed field, then its
cardinality is the same as that of a transcendence basis.

For a simpler, but less universe-polymorphic statement, see
`IsAlgClosed.cardinal_eq_cardinal_transcendence_basis_of_aleph0_lt'` -/
theorem cardinal_eq_cardinal_transcendence_basis_of_aleph0_lt [Nontrivial R]
    (hv : IsTranscendenceBasis R v) (hR : #R ≤ ℵ₀) (hK : ℵ₀ < #K) :
    Cardinal.lift.{w} #K = Cardinal.lift.{v} #ι :=
  have : ℵ₀ ≤ Cardinal.lift.{max u v} #ι := le_of_not_lt fun h => not_le_of_gt
                                             /-
                                               R : Type u
                                               K : Type v
                                               inst✝⁴ : CommRing R
                                               inst✝³ : Field K
                                               inst✝² : Algebra R K
                                               inst✝¹ : IsAlgClosed K
                                               ι : Type w
                                               v : ι → K
                                               inst✝ : Nontrivial R
                                               hv : IsTranscendenceBasis R v
                                               hR : LE.le (Cardinal.mk R) Cardinal.aleph0
                                               hK : LT.lt Cardinal.aleph0 (Cardinal.mk K)
                                               h : LT.lt (Cardinal.lift.{max u v, w} (Cardinal.mk ι)) Cardinal.aleph0
                                               ⊢ LT.lt Cardinal.aleph0 (Cardinal.lift.{max u w, v} (Cardinal.mk K))
                                             -/
    (show ℵ₀ < Cardinal.lift.{max u w} #K by simpa) <|
                                             /-
                                               🎉 no goals
                                             -/
    calc
      Cardinal.lift.{max u w, v} #K ≤ max (max (Cardinal.lift.{max v w, u} #R)
        (Cardinal.lift.{max u v, w} #ι)) ℵ₀ := cardinal_le_max_transcendence_basis v hv
                                  /-
                                    R : Type u
                                    K : Type v
                                    inst✝⁴ : CommRing R
                                    inst✝³ : Field K
                                    inst✝² : Algebra R K
                                    inst✝¹ : IsAlgClosed K
                                    ι : Type w
                                    v : ι → K
                                    inst✝ : Nontrivial R
                                    hv : IsTranscendenceBasis R v
                                    hR : LE.le (Cardinal.mk R) Cardinal.aleph0
                                    hK : LT.lt Cardinal.aleph0 (Cardinal.mk K)
                                    h : LT.lt (Cardinal.lift.{max u v, w} (Cardinal.mk ι)) Cardinal.aleph0
                                    ⊢ LE.le (Cardinal.lift.{max v w, u} (Cardinal.mk R)) Cardinal.aleph0
                                  -/
                                  /-
                                    🎉 no goals
                                  -/
      _ ≤ _ := max_le (max_le (by simpa) (by simpa using le_of_lt h)) le_rfl
                                             /-
                                               🎉 no goals
                                             -/
  suffices Cardinal.lift.{max u w} #K = Cardinal.lift.{max u v} #ι
                                                  /-
                                                    R : Type u
                                                    K : Type v
                                                    inst✝⁴ : CommRing R
                                                    inst✝³ : Field K
                                                    inst✝² : Algebra R K
                                                    inst✝¹ : IsAlgClosed K
                                                    ι : Type w
                                                    v : ι → K
                                                    inst✝ : Nontrivial R
                                                    hv : IsTranscendenceBasis R v
                                                    hR : LE.le (Cardinal.mk R) Cardinal.aleph0
                                                    hK : LT.lt Cardinal.aleph0 (Cardinal.mk K)
                                                    this✝ : LE.le Cardinal.aleph0 (Cardinal.lift.{max u v, w} (Cardinal.mk ι))
                                                    this : Eq (Cardinal.lift.{max u w, v} (Cardinal.mk K)) (Cardinal.lift.{max u v …
                                                    ⊢ Eq (Cardinal.lift.{u, max v w} (Cardinal.lift.{w, v} (Cardinal.mk K))) (Card …
                                                  -/
    from Cardinal.lift_injective.{u, max v w} (by simpa)
                                                  /-
                                                    🎉 no goals
                                                  -/
  le_antisymm
    (calc
      Cardinal.lift.{max u w} #K ≤ max (max
        (Cardinal.lift.{max v w} #R) (Cardinal.lift.{max u v} #ι)) ℵ₀ :=
        /-
          R : Type u
          K : Type v
          inst✝⁴ : CommRing R
          inst✝³ : Field K
          inst✝² : Algebra R K
          inst✝¹ : IsAlgClosed K
          ι : Type w
          v : ι → K
          inst✝ : Nontrivial R
          hv : IsTranscendenceBasis R v
          hR : LE.le (Cardinal.mk R) Cardinal.aleph0
          hK : LT.lt Cardinal.aleph0 (Cardinal.mk K)
          this : LE.le Cardinal.aleph0 (Cardinal.lift.{max u v, w} (Cardinal.mk ι))
          ⊢ Eq (Max.max (Max.max (Cardinal.lift.{max v w, u} (Cardinal.mk R)) (Cardinal. …
        -/
        cardinal_le_max_transcendence_basis v hv
          /-
            R : Type u
            K : Type v
            inst✝⁴ : CommRing R
            inst✝³ : Field K
            inst✝² : Algebra R K
            inst✝¹ : IsAlgClosed K
            ι : Type w
            v : ι → K
            inst✝ : Nontrivial R
            hv : IsTranscendenceBasis R v
            hR : LE.le (Cardinal.mk R) Cardinal.aleph0
            hK : LT.lt Cardinal.aleph0 (Cardinal.mk K)
            this : LE.le Cardinal.aleph0 (Cardinal.lift.{max u v, w} (Cardinal.mk ι))
            ⊢ LE.le (Cardinal.lift.{max v w, u} (Cardinal.mk R)) (Cardinal.lift.{max u v,  …
          -/
      _ = Cardinal.lift #ι := by
          /-
            🎉 no goals
          -/
          /-
            R : Type u
            K : Type v
            inst✝⁴ : CommRing R
            inst✝³ : Field K
            inst✝² : Algebra R K
            inst✝¹ : IsAlgClosed K
            ι : Type w
            v : ι → K
            inst✝ : Nontrivial R
            hv : IsTranscendenceBasis R v
            hR : LE.le (Cardinal.mk R) Cardinal.aleph0
            hK : LT.lt Cardinal.aleph0 (Cardinal.mk K)
            this : LE.le Cardinal.aleph0 (Cardinal.lift.{max u v, w} (Cardinal.mk ι))
            ⊢ LE.le Cardinal.aleph0 (Max.max (Cardinal.lift.{max v w, u} (Cardinal.mk R))  …
          -/
        rw [max_eq_left, max_eq_right]
          /-
            🎉 no goals
          -/
        · exact le_trans (by simpa using hR) this
        · exact le_max_of_le_right this)
    (lift_mk_le.2 ⟨⟨v, hv.1.injective⟩⟩)


/-- If `K` is an uncountable algebraically closed field, then its
cardinality is the same as that of a transcendence basis.

This is a simpler, but less general statement of
`cardinal_eq_cardinal_transcendence_basis_of_aleph0_lt`. -/
theorem cardinal_eq_cardinal_transcendence_basis_of_aleph0_lt' [Nontrivial R]
    (hv : IsTranscendenceBasis R v') (hR : #R ≤ ℵ₀) (hK : ℵ₀ < #K') : #K' = #ι' := by
  /-
    R : Type u
    inst✝⁴ : CommRing R
    K' : Type u
    inst✝³ : Field K'
    inst✝² : Algebra R K'
    inst✝¹ : IsAlgClosed K'
    ι' : Type u
    v' : ι' → K'
    inst✝ : Nontrivial R
    hv : IsTranscendenceBasis R v'
    hR : LE.le (Cardinal.mk R) Cardinal.aleph0
    hK : LT.lt Cardinal.aleph0 (Cardinal.mk K')
    ⊢ Eq (Cardinal.mk K') (Cardinal.mk ι')
  -/
  simpa using cardinal_eq_cardinal_transcendence_basis_of_aleph0_lt v' hv hR hK
  /-
    🎉 no goals
  -/


/-- Two uncountable algebraically closed fields of characteristic zero are isomorphic
if they have the same cardinality. -/
theorem ringEquiv_of_equiv_of_charZero [CharZero K] [CharZero L] (hK : ℵ₀ < #K)
    (hKL : Nonempty (K ≃ L)) : Nonempty (K ≃+* L) := by
  cases' exists_isTranscendenceBasis ℤ
    (show Function.Injective (algebraMap ℤ K) from Int.cast_injective) with s hs
  cases' exists_isTranscendenceBasis ℤ
    (show Function.Injective (algebraMap ℤ L) from Int.cast_injective) with t ht
  have hL : ℵ₀ < #L := by
    rwa [← aleph0_lt_lift.{v, u}, ← lift_mk_eq'.2 hKL, aleph0_lt_lift]
  have : Cardinal.lift.{v} #s = Cardinal.lift.{u} #t := by
    rw [← lift_injective (cardinal_eq_cardinal_transcendence_basis_of_aleph0_lt _
        hs (le_of_eq mk_int) hK),
      ← lift_injective (cardinal_eq_cardinal_transcendence_basis_of_aleph0_lt _
        ht (le_of_eq mk_int) hL)]
    exact Cardinal.lift_mk_eq'.2 hKL
  /-
    case intro.intro
    K : Type u
    L : Type v
    inst✝⁵ : Field K
    inst✝⁴ : Field L
    inst✝³ : IsAlgClosed K
    inst✝² : IsAlgClosed L
    inst✝¹ : CharZero K
    inst✝ : CharZero L
    hK : LT.lt Cardinal.aleph0 (Cardinal.mk K)
    hKL : Nonempty (Equiv K L)
    s : Set K
    hs : IsTranscendenceBasis Int Subtype.val
    t : Set L
    ht : IsTranscendenceBasis Int Subtype.val
    hL : LT.lt Cardinal.aleph0 (Cardinal.mk L)
    this : Eq (Cardinal.lift.{v, u} (Cardinal.mk ↑s)) (Cardinal.lift.{u, v} (Cardi …
    ⊢ Nonempty (RingEquiv K L)
  -/
  cases' Cardinal.lift_mk_eq'.1 this with e
  /-
    case intro.intro.intro
    K : Type u
    L : Type v
    inst✝⁵ : Field K
    inst✝⁴ : Field L
    inst✝³ : IsAlgClosed K
    inst✝² : IsAlgClosed L
    inst✝¹ : CharZero K
    inst✝ : CharZero L
    hK : LT.lt Cardinal.aleph0 (Cardinal.mk K)
    hKL : Nonempty (Equiv K L)
    s : Set K
    hs : IsTranscendenceBasis Int Subtype.val
    t : Set L
    ht : IsTranscendenceBasis Int Subtype.val
    hL : LT.lt Cardinal.aleph0 (Cardinal.mk L)
    this : Eq (Cardinal.lift.{v, u} (Cardinal.mk ↑s)) (Cardinal.lift.{u, v} (Cardi …
    e : Equiv ↑s ↑t
    ⊢ Nonempty (RingEquiv K L)
  -/
  exact ⟨equivOfTranscendenceBasis _ _ e hs ht⟩
  /-
    🎉 no goals
  -/


private theorem ringEquiv_of_Cardinal_eq_of_charP (p : ℕ) [Fact p.Prime] [CharP K p] [CharP L p]
    (hK : ℵ₀ < #K) (hKL : Nonempty (K ≃ L)) : Nonempty (K ≃+* L) := by
  /-
    K : Type u
    L : Type v
    inst✝⁶ : Field K
    inst✝⁵ : Field L
    inst✝⁴ : IsAlgClosed K
    inst✝³ : IsAlgClosed L
    p : Nat
    inst✝² : Fact (Nat.Prime p)
    inst✝¹ : CharP K p
    inst✝ : CharP L p
    hK : LT.lt Cardinal.aleph0 (Cardinal.mk K)
    hKL : Nonempty (Equiv K L)
    ⊢ Nonempty (RingEquiv K L)
  -/
  letI : Algebra (ZMod p) K := ZMod.algebra _ _
  /-
    K : Type u
    L : Type v
    inst✝⁶ : Field K
    inst✝⁵ : Field L
    inst✝⁴ : IsAlgClosed K
    inst✝³ : IsAlgClosed L
    p : Nat
    inst✝² : Fact (Nat.Prime p)
    inst✝¹ : CharP K p
    inst✝ : CharP L p
    hK : LT.lt Cardinal.aleph0 (Cardinal.mk K)
    hKL : Nonempty (Equiv K L)
    this : Algebra (ZMod p) K := ZMod.algebra K p
    ⊢ Nonempty (RingEquiv K L)
  -/
  letI : Algebra (ZMod p) L := ZMod.algebra _ _
  cases' exists_isTranscendenceBasis (ZMod p)
    (show Function.Injective (algebraMap (ZMod p) K) from RingHom.injective _) with s hs
  cases' exists_isTranscendenceBasis (ZMod p)
    (show Function.Injective (algebraMap (ZMod p) L) from RingHom.injective _) with t ht
  have hL : ℵ₀ < #L := by
    rwa [← aleph0_lt_lift.{v, u}, ← lift_mk_eq'.2 hKL, aleph0_lt_lift]
  have : Cardinal.lift.{v} #s = Cardinal.lift.{u} #t := by
    rw [← lift_injective (cardinal_eq_cardinal_transcendence_basis_of_aleph0_lt _
        hs (le_of_lt (lt_aleph0_of_finite _)) hK),
      ← lift_injective (cardinal_eq_cardinal_transcendence_basis_of_aleph0_lt _
        ht (le_of_lt (lt_aleph0_of_finite _)) hL)]
    exact Cardinal.lift_mk_eq'.2 hKL
  /-
    case intro.intro
    K : Type u
    L : Type v
    inst✝⁶ : Field K
    inst✝⁵ : Field L
    inst✝⁴ : IsAlgClosed K
    inst✝³ : IsAlgClosed L
    p : Nat
    inst✝² : Fact (Nat.Prime p)
    inst✝¹ : CharP K p
    inst✝ : CharP L p
    hK : LT.lt Cardinal.aleph0 (Cardinal.mk K)
    hKL : Nonempty (Equiv K L)
    this✝¹ : Algebra (ZMod p) K := ZMod.algebra K p
    this✝ : Algebra (ZMod p) L := ZMod.algebra L p
    s : Set K
    hs : IsTranscendenceBasis (ZMod p) Subtype.val
    t : Set L
    ht : IsTranscendenceBasis (ZMod p) Subtype.val
    hL : LT.lt Cardinal.aleph0 (Cardinal.mk L)
    this : Eq (Cardinal.lift.{v, u} (Cardinal.mk ↑s)) (Cardinal.lift.{u, v} (Cardi …
    ⊢ Nonempty (RingEquiv K L)
  -/
  cases' Cardinal.lift_mk_eq'.1 this with e
  /-
    case intro.intro.intro
    K : Type u
    L : Type v
    inst✝⁶ : Field K
    inst✝⁵ : Field L
    inst✝⁴ : IsAlgClosed K
    inst✝³ : IsAlgClosed L
    p : Nat
    inst✝² : Fact (Nat.Prime p)
    inst✝¹ : CharP K p
    inst✝ : CharP L p
    hK : LT.lt Cardinal.aleph0 (Cardinal.mk K)
    hKL : Nonempty (Equiv K L)
    this✝¹ : Algebra (ZMod p) K := ZMod.algebra K p
    this✝ : Algebra (ZMod p) L := ZMod.algebra L p
    s : Set K
    hs : IsTranscendenceBasis (ZMod p) Subtype.val
    t : Set L
    ht : IsTranscendenceBasis (ZMod p) Subtype.val
    hL : LT.lt Cardinal.aleph0 (Cardinal.mk L)
    this : Eq (Cardinal.lift.{v, u} (Cardinal.mk ↑s)) (Cardinal.lift.{u, v} (Cardi …
    e : Equiv ↑s ↑t
    ⊢ Nonempty (RingEquiv K L)
  -/
  exact ⟨equivOfTranscendenceBasis _ _ e hs ht⟩
  /-
    🎉 no goals
  -/


/-- Two uncountable algebraically closed fields are isomorphic
if they have the same cardinality and the same characteristic. -/
theorem ringEquiv_of_equiv_of_char_eq (p : ℕ) [CharP K p] [CharP L p] (hK : ℵ₀ < #K)
    (hKL : Nonempty (K ≃ L)) : Nonempty (K ≃+* L) := by
  /-
    K : Type u
    L : Type v
    inst✝⁵ : Field K
    inst✝⁴ : Field L
    inst✝³ : IsAlgClosed K
    inst✝² : IsAlgClosed L
    p : Nat
    inst✝¹ : CharP K p
    inst✝ : CharP L p
    hK : LT.lt Cardinal.aleph0 (Cardinal.mk K)
    hKL : Nonempty (Equiv K L)
    ⊢ Nonempty (RingEquiv K L)
  -/
  rcases CharP.char_is_prime_or_zero K p with (hp | hp)
    /-
      case inl
      K : Type u
      L : Type v
      inst✝⁵ : Field K
      inst✝⁴ : Field L
      inst✝³ : IsAlgClosed K
      inst✝² : IsAlgClosed L
      p : Nat
      inst✝¹ : CharP K p
      inst✝ : CharP L p
      hK : LT.lt Cardinal.aleph0 (Cardinal.mk K)
      hKL : Nonempty (Equiv K L)
      hp : Nat.Prime p
      ⊢ Nonempty (RingEquiv K L)
    -/
  · haveI : Fact p.Prime := ⟨hp⟩
    /-
      case inl
      K : Type u
      L : Type v
      inst✝⁵ : Field K
      inst✝⁴ : Field L
      inst✝³ : IsAlgClosed K
      inst✝² : IsAlgClosed L
      p : Nat
      inst✝¹ : CharP K p
      inst✝ : CharP L p
      hK : LT.lt Cardinal.aleph0 (Cardinal.mk K)
      hKL : Nonempty (Equiv K L)
      hp : Nat.Prime p
      this : Fact (Nat.Prime p)
      ⊢ Nonempty (RingEquiv K L)
    -/
    exact ringEquiv_of_Cardinal_eq_of_charP p hK hKL
    /-
      🎉 no goals
    -/
    /-
      case inr
      K : Type u
      L : Type v
      inst✝⁵ : Field K
      inst✝⁴ : Field L
      inst✝³ : IsAlgClosed K
      inst✝² : IsAlgClosed L
      p : Nat
      inst✝¹ : CharP K p
      inst✝ : CharP L p
      hK : LT.lt Cardinal.aleph0 (Cardinal.mk K)
      hKL : Nonempty (Equiv K L)
      hp : Eq p 0
      ⊢ Nonempty (RingEquiv K L)
    -/
  · simp only [hp] at *
    /-
      case inr
      K : Type u
      L : Type v
      inst✝⁵ : Field K
      inst✝⁴ : Field L
      inst✝³ : IsAlgClosed K
      inst✝² : IsAlgClosed L
      p : Nat
      hK : LT.lt Cardinal.aleph0 (Cardinal.mk K)
      hKL : Nonempty (Equiv K L)
      inst✝¹ : CharP K 0
      inst✝ : CharP L 0
      hp : True
      ⊢ Nonempty (RingEquiv K L)
    -/
    letI : CharZero K := CharP.charP_to_charZero K
    /-
      case inr
      K : Type u
      L : Type v
      inst✝⁵ : Field K
      inst✝⁴ : Field L
      inst✝³ : IsAlgClosed K
      inst✝² : IsAlgClosed L
      p : Nat
      hK : LT.lt Cardinal.aleph0 (Cardinal.mk K)
      hKL : Nonempty (Equiv K L)
      inst✝¹ : CharP K 0
      inst✝ : CharP L 0
      hp : True
      this : CharZero K := CharP.charP_to_charZero K
      ⊢ Nonempty (RingEquiv K L)
    -/
    letI : CharZero L := CharP.charP_to_charZero L
    /-
      case inr
      K : Type u
      L : Type v
      inst✝⁵ : Field K
      inst✝⁴ : Field L
      inst✝³ : IsAlgClosed K
      inst✝² : IsAlgClosed L
      p : Nat
      hK : LT.lt Cardinal.aleph0 (Cardinal.mk K)
      hKL : Nonempty (Equiv K L)
      inst✝¹ : CharP K 0
      inst✝ : CharP L 0
      hp : True
      this✝ : CharZero K := CharP.charP_to_charZero K
      this : CharZero L := CharP.charP_to_charZero L
      ⊢ Nonempty (RingEquiv K L)
    -/
    exact ringEquiv_of_equiv_of_charZero hK hKL
    /-
      🎉 no goals
    -/


