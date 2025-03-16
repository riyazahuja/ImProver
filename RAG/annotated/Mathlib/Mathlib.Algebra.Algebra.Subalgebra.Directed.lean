theorem coe_iSup_of_directed (dir : Directed (· ≤ ·) K) : ↑(iSup K) = ⋃ i, (K i : Set A) :=
  let s : Subalgebra R A :=
    { __ := Subsemiring.copy _ _ (Subsemiring.coe_iSup_of_directed dir).symm
      algebraMap_mem' := fun _ ↦ Set.mem_iUnion.2
        ⟨Classical.arbitrary ι, Subalgebra.algebraMap_mem _ _⟩ }
  have : iSup K = s := le_antisymm
    (iSup_le fun i ↦ le_iSup (fun i ↦ (K i : Set A)) i) (Set.iUnion_subset fun _ ↦ le_iSup K _)
  this.symm ▸ rfl


/-- Define an algebra homomorphism on a directed supremum of subalgebras by defining
it on each subalgebra, and proving that it agrees on the intersection of subalgebras. -/
noncomputable def iSupLift (dir : Directed (· ≤ ·) K) (f : ∀ i, K i →ₐ[R] B)
    (hf : ∀ (i j : ι) (h : K i ≤ K j), f i = (f j).comp (inclusion h))
    (T : Subalgebra R A) (hT : T = iSup K): ↥T →ₐ[R] B :=
  { toFun := Set.iUnionLift (fun i => ↑(K i)) (fun i x => f i x)
        (fun i j x hxi hxj => by
          /-
            R : Type u_1
            A : Type u_2
            B : Type u_3
            inst✝⁵ : CommSemiring R
            inst✝⁴ : Semiring A
            inst✝³ : Algebra R A
            inst✝² : Semiring B
            inst✝¹ : Algebra R B
            S : Subalgebra R A
            ι : Type u_4
            inst✝ : Nonempty ι
            K : ι → Subalgebra R A
            dir : Directed (fun x1 x2 => LE.le x1 x2) K
            f : (i : ι) → AlgHom R (Subtype fun x => Membership.mem (K i) x) B
            hf : ∀ (i j : ι) (h : LE.le (K i) (K j)), Eq (f i) ((f j).comp (Subalgebra.inc …
            T : Subalgebra R A
            hT : Eq T (iSup K)
            i j : ι
            x : A
            hxi : Membership.mem ((fun i => ↑(K i)) i) x
            hxj : Membership.mem ((fun i => ↑(K i)) j) x
            ⊢ Eq ((fun i x => (f i) x) i ⟨x, hxi⟩) ((fun i x => (f i) x) j ⟨x, hxj⟩)
          -/
          let ⟨k, hik, hjk⟩ := dir i j
          /-
            R : Type u_1
            A : Type u_2
            B : Type u_3
            inst✝⁵ : CommSemiring R
            inst✝⁴ : Semiring A
            inst✝³ : Algebra R A
            inst✝² : Semiring B
            inst✝¹ : Algebra R B
            S : Subalgebra R A
            ι : Type u_4
            inst✝ : Nonempty ι
            K : ι → Subalgebra R A
            dir : Directed (fun x1 x2 => LE.le x1 x2) K
            f : (i : ι) → AlgHom R (Subtype fun x => Membership.mem (K i) x) B
            hf : ∀ (i j : ι) (h : LE.le (K i) (K j)), Eq (f i) ((f j).comp (Subalgebra.inc …
            T : Subalgebra R A
            hT : Eq T (iSup K)
            i j : ι
            x : A
            hxi : Membership.mem ((fun i => ↑(K i)) i) x
            hxj : Membership.mem ((fun i => ↑(K i)) j) x
            k : ι
            hik : (fun x1 x2 => LE.le x1 x2) (K i) (K k)
            hjk : (fun x1 x2 => LE.le x1 x2) (K j) (K k)
            ⊢ Eq ((fun i x => (f i) x) i ⟨x, hxi⟩) ((fun i x => (f i) x) j ⟨x, hxj⟩)
          -/
          dsimp
          /-
            R : Type u_1
            A : Type u_2
            B : Type u_3
            inst✝⁵ : CommSemiring R
            inst✝⁴ : Semiring A
            inst✝³ : Algebra R A
            inst✝² : Semiring B
            inst✝¹ : Algebra R B
            S : Subalgebra R A
            ι : Type u_4
            inst✝ : Nonempty ι
            K : ι → Subalgebra R A
            dir : Directed (fun x1 x2 => LE.le x1 x2) K
            f : (i : ι) → AlgHom R (Subtype fun x => Membership.mem (K i) x) B
            hf : ∀ (i j : ι) (h : LE.le (K i) (K j)), Eq (f i) ((f j).comp (Subalgebra.inc …
            T : Subalgebra R A
            hT : Eq T (iSup K)
            i j : ι
            x : A
            hxi : Membership.mem ((fun i => ↑(K i)) i) x
            hxj : Membership.mem ((fun i => ↑(K i)) j) x
            k : ι
            hik : (fun x1 x2 => LE.le x1 x2) (K i) (K k)
            hjk : (fun x1 x2 => LE.le x1 x2) (K j) (K k)
            ⊢ Eq ((f i) ⟨x, hxi⟩) ((f j) ⟨x, hxj⟩)
          -/
          rw [hf i k hik, hf j k hjk]
          /-
            R : Type u_1
            A : Type u_2
            B : Type u_3
            inst✝⁵ : CommSemiring R
            inst✝⁴ : Semiring A
            inst✝³ : Algebra R A
            inst✝² : Semiring B
            inst✝¹ : Algebra R B
            S : Subalgebra R A
            ι : Type u_4
            inst✝ : Nonempty ι
            K : ι → Subalgebra R A
            dir : Directed (fun x1 x2 => LE.le x1 x2) K
            f : (i : ι) → AlgHom R (Subtype fun x => Membership.mem (K i) x) B
            hf : ∀ (i j : ι) (h : LE.le (K i) (K j)), Eq (f i) ((f j).comp (Subalgebra.inc …
            T : Subalgebra R A
            hT : Eq T (iSup K)
            i j : ι
            x : A
            hxi : Membership.mem ((fun i => ↑(K i)) i) x
            hxj : Membership.mem ((fun i => ↑(K i)) j) x
            k : ι
            hik : (fun x1 x2 => LE.le x1 x2) (K i) (K k)
            hjk : (fun x1 x2 => LE.le x1 x2) (K j) (K k)
            ⊢ Eq (((f k).comp (Subalgebra.inclusion hik)) ⟨x, hxi⟩) (((f k).comp (Subalgeb …
          -/
          rfl)
          /-
            🎉 no goals
          -/
                        /-
                          R : Type u_1
                          A : Type u_2
                          B : Type u_3
                          inst✝⁵ : CommSemiring R
                          inst✝⁴ : Semiring A
                          inst✝³ : Algebra R A
                          inst✝² : Semiring B
                          inst✝¹ : Algebra R B
                          S : Subalgebra R A
                          ι : Type u_4
                          inst✝ : Nonempty ι
                          K : ι → Subalgebra R A
                          dir : Directed (fun x1 x2 => LE.le x1 x2) K
                          f : (i : ι) → AlgHom R (Subtype fun x => Membership.mem (K i) x) B
                          hf : ∀ (i j : ι) (h : LE.le (K i) (K j)), Eq (f i) ((f j).comp (Subalgebra.inc …
                          T : Subalgebra R A
                          hT : Eq T (iSup K)
                          ⊢ HasSubset.Subset (↑T) (Set.iUnion fun i => ↑(K i))
                        -/
        (T : Set A) (by rw [hT, coe_iSup_of_directed dir])
                        /-
                          🎉 no goals
                        -/
                   /-
                     R : Type u_1
                     A : Type u_2
                     B : Type u_3
                     inst✝⁵ : CommSemiring R
                     inst✝⁴ : Semiring A
                     inst✝³ : Algebra R A
                     inst✝² : Semiring B
                     inst✝¹ : Algebra R B
                     S : Subalgebra R A
                     ι : Type u_4
                     inst✝ : Nonempty ι
                     K : ι → Subalgebra R A
                     dir : Directed (fun x1 x2 => LE.le x1 x2) K
                     f : (i : ι) → AlgHom R (Subtype fun x => Membership.mem (K i) x) B
                     hf : ∀ (i j : ι) (h : LE.le (K i) (K j)), Eq (f i) ((f j).comp (Subalgebra.inc …
                     T : Subalgebra R A
                     hT : Eq T (iSup K)
                     ⊢ Eq (Set.iUnionLift (fun i => ↑(K i)) (fun i x => (f i) x) ⋯ ↑T ⋯ 1) 1
                   -/
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
    map_one' := by apply Set.iUnionLift_const _ (fun _ => 1) <;> simp
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
                    /-
                      R : Type u_1
                      A : Type u_2
                      B : Type u_3
                      inst✝⁵ : CommSemiring R
                      inst✝⁴ : Semiring A
                      inst✝³ : Algebra R A
                      inst✝² : Semiring B
                      inst✝¹ : Algebra R B
                      S : Subalgebra R A
                      ι : Type u_4
                      inst✝ : Nonempty ι
                      K : ι → Subalgebra R A
                      dir : Directed (fun x1 x2 => LE.le x1 x2) K
                      f : (i : ι) → AlgHom R (Subtype fun x => Membership.mem (K i) x) B
                      hf : ∀ (i j : ι) (h : LE.le (K i) (K j)), Eq (f i) ((f j).comp (Subalgebra.inc …
                      T : Subalgebra R A
                      hT : Eq T (iSup K)
                      ⊢ Eq ((↑{ toFun := Set.iUnionLift (fun i => ↑(K i)) (fun i x => (f i) x) ⋯ ↑T  …
                    -/
                                                                         /-
                                                                           🎉 no goals
                                                                         -/
      /-
        R : Type u_1
        A : Type u_2
        B : Type u_3
        inst✝⁵ : CommSemiring R
        inst✝⁴ : Semiring A
        inst✝³ : Algebra R A
        inst✝² : Semiring B
        inst✝¹ : Algebra R B
        S : Subalgebra R A
        ι : Type u_4
        inst✝ : Nonempty ι
        K : ι → Subalgebra R A
        dir : Directed (fun x1 x2 => LE.le x1 x2) K
        f : (i : ι) → AlgHom R (Subtype fun x => Membership.mem (K i) x) B
        hf : ∀ (i j : ι) (h : LE.le (K i) (K j)), Eq (f i) ((f j).comp (Subalgebra.inc …
        T : Subalgebra R A
        hT : Eq T (iSup K)
        ⊢ ∀ (x y : Subtype fun x => Membership.mem T x), Eq ({ toFun := Set.iUnionLift …
      -/
    map_zero' := by dsimp; apply Set.iUnionLift_const _ (fun _ => 0) <;> simp
      /-
        R : Type u_1
        A : Type u_2
        B : Type u_3
        inst✝⁵ : CommSemiring R
        inst✝⁴ : Semiring A
        inst✝³ : Algebra R A
        inst✝² : Semiring B
        inst✝¹ : Algebra R B
        S : Subalgebra R A
        ι : Type u_4
        inst✝ : Nonempty ι
        K : ι → Subalgebra R A
        dir : Directed (fun x1 x2 => LE.le x1 x2) K
        f : (i : ι) → AlgHom R (Subtype fun x => Membership.mem (K i) x) B
        hf : ∀ (i j : ι) (h : LE.le (K i) (K j)), Eq (f i) ((f j).comp (Subalgebra.inc …
        ⊢ ∀ (x y : Subtype fun x => Membership.mem (iSup K) x), Eq (Set.iUnionLift (fu …
      -/
                                                                         /-
                                                                           🎉 no goals
                                                                         -/
      /-
        case hopi
        R : Type u_1
        A : Type u_2
        B : Type u_3
        inst✝⁵ : CommSemiring R
        inst✝⁴ : Semiring A
        inst✝³ : Algebra R A
        inst✝² : Semiring B
        inst✝¹ : Algebra R B
        S : Subalgebra R A
        ι : Type u_4
        inst✝ : Nonempty ι
        K : ι → Subalgebra R A
        dir : Directed (fun x1 x2 => LE.le x1 x2) K
        f : (i : ι) → AlgHom R (Subtype fun x => Membership.mem (K i) x) B
        hf : ∀ (i j : ι) (h : LE.le (K i) (K j)), Eq (f i) ((f j).comp (Subalgebra.inc …
        ⊢ ∀ (i : ι) (x y : ↑↑(K i)), Eq (Set.inclusion ⋯ (HMul.hMul x y)) (HMul.hMul ( …
      -/
    map_mul' := by
      /-
        🎉 no goals
      -/
      subst hT; dsimp
      apply Set.iUnionLift_binary (coe_iSup_of_directed dir) dir _ (fun _ => (· * ·))
      all_goals simp
    map_add' := by
      /-
        R : Type u_1
        A : Type u_2
        B : Type u_3
        inst✝⁵ : CommSemiring R
        inst✝⁴ : Semiring A
        inst✝³ : Algebra R A
        inst✝² : Semiring B
        inst✝¹ : Algebra R B
        S : Subalgebra R A
        ι : Type u_4
        inst✝ : Nonempty ι
        K : ι → Subalgebra R A
        dir : Directed (fun x1 x2 => LE.le x1 x2) K
        f : (i : ι) → AlgHom R (Subtype fun x => Membership.mem (K i) x) B
        hf : ∀ (i j : ι) (h : LE.le (K i) (K j)), Eq (f i) ((f j).comp (Subalgebra.inc …
        T : Subalgebra R A
        hT : Eq T (iSup K)
        ⊢ ∀ (x y : Subtype fun x => Membership.mem T x), Eq ((↑{ toFun := Set.iUnionLi …
      -/
      subst hT; dsimp
      /-
        R : Type u_1
        A : Type u_2
        B : Type u_3
        inst✝⁵ : CommSemiring R
        inst✝⁴ : Semiring A
        inst✝³ : Algebra R A
        inst✝² : Semiring B
        inst✝¹ : Algebra R B
        S : Subalgebra R A
        ι : Type u_4
        inst✝ : Nonempty ι
        K : ι → Subalgebra R A
        dir : Directed (fun x1 x2 => LE.le x1 x2) K
        f : (i : ι) → AlgHom R (Subtype fun x => Membership.mem (K i) x) B
        hf : ∀ (i j : ι) (h : LE.le (K i) (K j)), Eq (f i) ((f j).comp (Subalgebra.inc …
        ⊢ ∀ (x y : Subtype fun x => Membership.mem (iSup K) x), Eq (Set.iUnionLift (fu …
      -/
      apply Set.iUnionLift_binary (coe_iSup_of_directed dir) dir _ (fun _ => (· + ·))
      /-
        case hopi
        R : Type u_1
        A : Type u_2
        B : Type u_3
        inst✝⁵ : CommSemiring R
        inst✝⁴ : Semiring A
        inst✝³ : Algebra R A
        inst✝² : Semiring B
        inst✝¹ : Algebra R B
        S : Subalgebra R A
        ι : Type u_4
        inst✝ : Nonempty ι
        K : ι → Subalgebra R A
        dir : Directed (fun x1 x2 => LE.le x1 x2) K
        f : (i : ι) → AlgHom R (Subtype fun x => Membership.mem (K i) x) B
        hf : ∀ (i j : ι) (h : LE.le (K i) (K j)), Eq (f i) ((f j).comp (Subalgebra.inc …
        ⊢ ∀ (i : ι) (x y : ↑↑(K i)), Eq (Set.inclusion ⋯ (HAdd.hAdd x y)) (HAdd.hAdd ( …
      -/
      all_goals simp
      /-
        🎉 no goals
      -/
    commutes' := fun r => by
      /-
        R : Type u_1
        A : Type u_2
        B : Type u_3
        inst✝⁵ : CommSemiring R
        inst✝⁴ : Semiring A
        inst✝³ : Algebra R A
        inst✝² : Semiring B
        inst✝¹ : Algebra R B
        S : Subalgebra R A
        ι : Type u_4
        inst✝ : Nonempty ι
        K : ι → Subalgebra R A
        dir : Directed (fun x1 x2 => LE.le x1 x2) K
        f : (i : ι) → AlgHom R (Subtype fun x => Membership.mem (K i) x) B
        hf : ∀ (i j : ι) (h : LE.le (K i) (K j)), Eq (f i) ((f j).comp (Subalgebra.inc …
        T : Subalgebra R A
        hT : Eq T (iSup K)
        r : R
        ⊢ Eq ((↑↑{ toFun := Set.iUnionLift (fun i => ↑(K i)) (fun i x => (f i) x) ⋯ ↑T …
      -/
      dsimp
      /-
        R : Type u_1
        A : Type u_2
        B : Type u_3
        inst✝⁵ : CommSemiring R
        inst✝⁴ : Semiring A
        inst✝³ : Algebra R A
        inst✝² : Semiring B
        inst✝¹ : Algebra R B
        S : Subalgebra R A
        ι : Type u_4
        inst✝ : Nonempty ι
        K : ι → Subalgebra R A
        dir : Directed (fun x1 x2 => LE.le x1 x2) K
        f : (i : ι) → AlgHom R (Subtype fun x => Membership.mem (K i) x) B
        hf : ∀ (i j : ι) (h : LE.le (K i) (K j)), Eq (f i) ((f j).comp (Subalgebra.inc …
        T : Subalgebra R A
        hT : Eq T (iSup K)
        r : R
        ⊢ Eq (Set.iUnionLift (fun i => ↑(K i)) (fun i x => (f i) x) ⋯ ↑T ⋯ ((algebraMa …
      -/
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
      apply Set.iUnionLift_const _ (fun _ => algebraMap R _ r) <;> simp }
                                                                   /-
                                                                     🎉 no goals
                                                                   -/



@[simp]
theorem iSupLift_inclusion {dir : Directed (· ≤ ·) K} {f : ∀ i, K i →ₐ[R] B}
    {hf : ∀ (i j : ι) (h : K i ≤ K j), f i = (f j).comp (inclusion h)}
    {T : Subalgebra R A} {hT : T = iSup K} {i : ι} (x : K i) (h : K i ≤ T) :
    iSupLift K dir f hf T hT (inclusion h x) = f i x := by
  /-
    R : Type u_1
    A : Type u_2
    B : Type u_3
    inst✝⁵ : CommSemiring R
    inst✝⁴ : Semiring A
    inst✝³ : Algebra R A
    inst✝² : Semiring B
    inst✝¹ : Algebra R B
    ι : Type u_4
    inst✝ : Nonempty ι
    K : ι → Subalgebra R A
    dir : Directed (fun x1 x2 => LE.le x1 x2) K
    f : (i : ι) → AlgHom R (Subtype fun x => Membership.mem (K i) x) B
    hf : ∀ (i j : ι) (h : LE.le (K i) (K j)), Eq (f i) ((f j).comp (Subalgebra.inc …
    T : Subalgebra R A
    hT : Eq T (iSup K)
    i : ι
    x : Subtype fun x => Membership.mem (K i) x
    h : LE.le (K i) T
    ⊢ Eq ((Subalgebra.iSupLift K dir f hf T hT) ((Subalgebra.inclusion h) x)) ((f  …
  -/
  dsimp [iSupLift, inclusion]
  /-
    R : Type u_1
    A : Type u_2
    B : Type u_3
    inst✝⁵ : CommSemiring R
    inst✝⁴ : Semiring A
    inst✝³ : Algebra R A
    inst✝² : Semiring B
    inst✝¹ : Algebra R B
    ι : Type u_4
    inst✝ : Nonempty ι
    K : ι → Subalgebra R A
    dir : Directed (fun x1 x2 => LE.le x1 x2) K
    f : (i : ι) → AlgHom R (Subtype fun x => Membership.mem (K i) x) B
    hf : ∀ (i j : ι) (h : LE.le (K i) (K j)), Eq (f i) ((f j).comp (Subalgebra.inc …
    T : Subalgebra R A
    hT : Eq T (iSup K)
    i : ι
    x : Subtype fun x => Membership.mem (K i) x
    h : LE.le (K i) T
    ⊢ Eq (Set.iUnionLift (fun i => ↑(K i)) (fun i x => (f i) x) ⋯ ↑T ⋯ (Set.inclus …
  -/
  rw [Set.iUnionLift_inclusion]
  /-
    🎉 no goals
  -/


@[simp]
theorem iSupLift_comp_inclusion {dir : Directed (· ≤ ·) K} {f : ∀ i, K i →ₐ[R] B}
    {hf : ∀ (i j : ι) (h : K i ≤ K j), f i = (f j).comp (inclusion h)}
    {T : Subalgebra R A} {hT : T = iSup K} {i : ι} (h : K i ≤ T) :
                                                              /-
                                                                R : Type u_1
                                                                A : Type u_2
                                                                B : Type u_3
                                                                inst✝⁵ : CommSemiring R
                                                                inst✝⁴ : Semiring A
                                                                inst✝³ : Algebra R A
                                                                inst✝² : Semiring B
                                                                inst✝¹ : Algebra R B
                                                                ι : Type u_4
                                                                inst✝ : Nonempty ι
                                                                K : ι → Subalgebra R A
                                                                dir : Directed (fun x1 x2 => LE.le x1 x2) K
                                                                f : (i : ι) → AlgHom R (Subtype fun x => Membership.mem (K i) x) B
                                                                hf : ∀ (i j : ι) (h : LE.le (K i) (K j)), Eq (f i) ((f j).comp (Subalgebra.inc …
                                                                T : Subalgebra R A
                                                                hT : Eq T (iSup K)
                                                                i : ι
                                                                h : LE.le (K i) T
                                                                ⊢ Eq ((Subalgebra.iSupLift K dir f hf T hT).comp (Subalgebra.inclusion h)) (f i)
                                                              -/
    (iSupLift K dir f hf T hT).comp (inclusion h) = f i := by ext; simp
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


@[simp]
theorem iSupLift_mk {dir : Directed (· ≤ ·) K} {f : ∀ i, K i →ₐ[R] B}
    {hf : ∀ (i j : ι) (h : K i ≤ K j), f i = (f j).comp (inclusion h)}
    {T : Subalgebra R A} {hT : T = iSup K} {i : ι} (x : K i) (hx : (x : A) ∈ T) :
    iSupLift K dir f hf T hT ⟨x, hx⟩ = f i x := by
  /-
    R : Type u_1
    A : Type u_2
    B : Type u_3
    inst✝⁵ : CommSemiring R
    inst✝⁴ : Semiring A
    inst✝³ : Algebra R A
    inst✝² : Semiring B
    inst✝¹ : Algebra R B
    ι : Type u_4
    inst✝ : Nonempty ι
    K : ι → Subalgebra R A
    dir : Directed (fun x1 x2 => LE.le x1 x2) K
    f : (i : ι) → AlgHom R (Subtype fun x => Membership.mem (K i) x) B
    hf : ∀ (i j : ι) (h : LE.le (K i) (K j)), Eq (f i) ((f j).comp (Subalgebra.inc …
    T : Subalgebra R A
    hT : Eq T (iSup K)
    i : ι
    x : Subtype fun x => Membership.mem (K i) x
    hx : Membership.mem T ↑x
    ⊢ Eq ((Subalgebra.iSupLift K dir f hf T hT) ⟨↑x, hx⟩) ((f i) x)
  -/
  dsimp [iSupLift, inclusion]
  /-
    R : Type u_1
    A : Type u_2
    B : Type u_3
    inst✝⁵ : CommSemiring R
    inst✝⁴ : Semiring A
    inst✝³ : Algebra R A
    inst✝² : Semiring B
    inst✝¹ : Algebra R B
    ι : Type u_4
    inst✝ : Nonempty ι
    K : ι → Subalgebra R A
    dir : Directed (fun x1 x2 => LE.le x1 x2) K
    f : (i : ι) → AlgHom R (Subtype fun x => Membership.mem (K i) x) B
    hf : ∀ (i j : ι) (h : LE.le (K i) (K j)), Eq (f i) ((f j).comp (Subalgebra.inc …
    T : Subalgebra R A
    hT : Eq T (iSup K)
    i : ι
    x : Subtype fun x => Membership.mem (K i) x
    hx : Membership.mem T ↑x
    ⊢ Eq (Set.iUnionLift (fun i => ↑(K i)) (fun i x => (f i) x) ⋯ ↑T ⋯ ⟨↑x, hx⟩) ( …
  -/
  rw [Set.iUnionLift_mk]
  /-
    🎉 no goals
  -/


theorem iSupLift_of_mem {dir : Directed (· ≤ ·) K} {f : ∀ i, K i →ₐ[R] B}
    {hf : ∀ (i j : ι) (h : K i ≤ K j), f i = (f j).comp (inclusion h)}
    {T : Subalgebra R A} {hT : T = iSup K} {i : ι} (x : T) (hx : (x : A) ∈ K i) :
    iSupLift K dir f hf T hT x = f i ⟨x, hx⟩ := by
  /-
    R : Type u_1
    A : Type u_2
    B : Type u_3
    inst✝⁵ : CommSemiring R
    inst✝⁴ : Semiring A
    inst✝³ : Algebra R A
    inst✝² : Semiring B
    inst✝¹ : Algebra R B
    ι : Type u_4
    inst✝ : Nonempty ι
    K : ι → Subalgebra R A
    dir : Directed (fun x1 x2 => LE.le x1 x2) K
    f : (i : ι) → AlgHom R (Subtype fun x => Membership.mem (K i) x) B
    hf : ∀ (i j : ι) (h : LE.le (K i) (K j)), Eq (f i) ((f j).comp (Subalgebra.inc …
    T : Subalgebra R A
    hT : Eq T (iSup K)
    i : ι
    x : Subtype fun x => Membership.mem T x
    hx : Membership.mem (K i) ↑x
    ⊢ Eq ((Subalgebra.iSupLift K dir f hf T hT) x) ((f i) ⟨↑x, hx⟩)
  -/
  dsimp [iSupLift, inclusion]
  /-
    R : Type u_1
    A : Type u_2
    B : Type u_3
    inst✝⁵ : CommSemiring R
    inst✝⁴ : Semiring A
    inst✝³ : Algebra R A
    inst✝² : Semiring B
    inst✝¹ : Algebra R B
    ι : Type u_4
    inst✝ : Nonempty ι
    K : ι → Subalgebra R A
    dir : Directed (fun x1 x2 => LE.le x1 x2) K
    f : (i : ι) → AlgHom R (Subtype fun x => Membership.mem (K i) x) B
    hf : ∀ (i j : ι) (h : LE.le (K i) (K j)), Eq (f i) ((f j).comp (Subalgebra.inc …
    T : Subalgebra R A
    hT : Eq T (iSup K)
    i : ι
    x : Subtype fun x => Membership.mem T x
    hx : Membership.mem (K i) ↑x
    ⊢ Eq (Set.iUnionLift (fun i => ↑(K i)) (fun i x => (f i) x) ⋯ ↑T ⋯ x) ((f i) ⟨ …
  -/
  rw [Set.iUnionLift_of_mem]
  /-
    🎉 no goals
  -/


