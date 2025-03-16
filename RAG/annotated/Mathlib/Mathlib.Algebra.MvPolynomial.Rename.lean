/-- Rename all the variables in a multivariable polynomial. -/
def rename (f : σ → τ) : MvPolynomial σ R →ₐ[R] MvPolynomial τ R :=
  aeval (X ∘ f)


theorem rename_C (f : σ → τ) (r : R) : rename f (C r) = C r :=
  eval₂_C _ _ _


@[simp]
theorem rename_X (f : σ → τ) (i : σ) : rename f (X i : MvPolynomial σ R) = X (f i) :=
  eval₂_X _ _ _


theorem map_rename (f : R →+* S) (g : σ → τ) (p : MvPolynomial σ R) :
    map f (rename g p) = rename g (map f p) := by
  apply MvPolynomial.induction_on p
    (fun a => by simp only [map_C, rename_C])
    (fun p q hp hq => by simp only [hp, hq, map_add]) fun p n hp => by
    simp only [hp, rename_X, map_X, map_mul]


@[simp]
theorem rename_rename (f : σ → τ) (g : τ → α) (p : MvPolynomial σ R) :
    rename g (rename f p) = rename (g ∘ f) p :=
  show rename g (eval₂ C (X ∘ f) p) = _ by
    /-
      σ : Type u_1
      τ : Type u_2
      α : Type u_3
      R : Type u_4
      inst✝ : CommSemiring R
      f : σ → τ
      g : τ → α
      p : MvPolynomial σ R
      ⊢ Eq ((MvPolynomial.rename g) (MvPolynomial.eval₂ MvPolynomial.C (Function.com …
    -/
    simp only [rename, aeval_eq_eval₂Hom]
    -- Porting note: the Lean 3 proof of this was very fragile and included a nonterminal `simp`.
    -- Hopefully this is less prone to breaking
    /-
      σ : Type u_1
      τ : Type u_2
      α : Type u_3
      R : Type u_4
      inst✝ : CommSemiring R
      f : σ → τ
      g : τ → α
      p : MvPolynomial σ R
      ⊢ Eq ((MvPolynomial.eval₂Hom (algebraMap R (MvPolynomial α R)) (Function.comp  …
    -/
    rw [eval₂_comp_left (eval₂Hom (algebraMap R (MvPolynomial α R)) (X ∘ g)) C (X ∘ f) p]
    /-
      σ : Type u_1
      τ : Type u_2
      α : Type u_3
      R : Type u_4
      inst✝ : CommSemiring R
      f : σ → τ
      g : τ → α
      p : MvPolynomial σ R
      ⊢ Eq (MvPolynomial.eval₂ ((MvPolynomial.eval₂Hom (algebraMap R (MvPolynomial α …
    -/
    simp only [comp_def, eval₂Hom_X']
    /-
      σ : Type u_1
      τ : Type u_2
      α : Type u_3
      R : Type u_4
      inst✝ : CommSemiring R
      f : σ → τ
      g : τ → α
      p : MvPolynomial σ R
      ⊢ Eq (MvPolynomial.eval₂ ((MvPolynomial.eval₂Hom (algebraMap R (MvPolynomial α …
    -/
    refine eval₂Hom_congr ?_ rfl rfl
    /-
      σ : Type u_1
      τ : Type u_2
      α : Type u_3
      R : Type u_4
      inst✝ : CommSemiring R
      f : σ → τ
      g : τ → α
      p : MvPolynomial σ R
      ⊢ Eq ((MvPolynomial.eval₂Hom (algebraMap R (MvPolynomial α R)) fun x => MvPoly …
    -/
    ext1; simp only [comp_apply, RingHom.coe_comp, eval₂Hom_C]
          /-
            🎉 no goals
          -/


@[simp]
theorem rename_id (p : MvPolynomial σ R) : rename id p = p :=
  eval₂_eta p


theorem rename_monomial (f : σ → τ) (d : σ →₀ ℕ) (r : R) :
    rename f (monomial d r) = monomial (d.mapDomain f) r := by
  rw [rename, aeval_monomial, monomial_eq (s := Finsupp.mapDomain f d),
    Finsupp.prod_mapDomain_index]
    /-
      σ : Type u_1
      τ : Type u_2
      R : Type u_4
      inst✝ : CommSemiring R
      f : σ → τ
      d : Finsupp σ Nat
      r : R
      ⊢ Eq (HMul.hMul ((algebraMap R (MvPolynomial τ R)) r) (d.prod fun i k => HPow. …
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case h_zero
      σ : Type u_1
      τ : Type u_2
      R : Type u_4
      inst✝ : CommSemiring R
      f : σ → τ
      d : Finsupp σ Nat
      r : R
      ⊢ ∀ (b : τ), Eq (HPow.hPow (MvPolynomial.X b) 0) 1
    -/
  · exact fun n => pow_zero _
    /-
      🎉 no goals
    -/
    /-
      case h_add
      σ : Type u_1
      τ : Type u_2
      R : Type u_4
      inst✝ : CommSemiring R
      f : σ → τ
      d : Finsupp σ Nat
      r : R
      ⊢ ∀ (b : τ) (m₁ m₂ : Nat), Eq (HPow.hPow (MvPolynomial.X b) (HAdd.hAdd m₁ m₂)) …
    -/
  · exact fun n i₁ i₂ => pow_add _ _ _
    /-
      🎉 no goals
    -/


theorem rename_eq (f : σ → τ) (p : MvPolynomial σ R) :
    rename f p = Finsupp.mapDomain (Finsupp.mapDomain f) p := by
  simp only [rename, aeval_def, eval₂, Finsupp.mapDomain, algebraMap_eq, comp_apply,
    X_pow_eq_monomial, ← monomial_finsupp_sum_index]
  /-
    σ : Type u_1
    τ : Type u_2
    R : Type u_4
    inst✝ : CommSemiring R
    f : σ → τ
    p : MvPolynomial σ R
    ⊢ Eq (Finsupp.sum p fun s a => (MvPolynomial.monomial (s.sum fun a => Finsupp. …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem rename_injective (f : σ → τ) (hf : Function.Injective f) :
    Function.Injective (rename f : MvPolynomial σ R → MvPolynomial τ R) := by
  have :
    (rename f : MvPolynomial σ R → MvPolynomial τ R) = Finsupp.mapDomain (Finsupp.mapDomain f) :=
    funext (rename_eq f)
  /-
    σ : Type u_1
    τ : Type u_2
    R : Type u_4
    inst✝ : CommSemiring R
    f : σ → τ
    hf : Function.Injective f
    this : Eq (⇑(MvPolynomial.rename f)) (Finsupp.mapDomain (Finsupp.mapDomain f))
    ⊢ Function.Injective ⇑(MvPolynomial.rename f)
  -/
  rw [this]
  /-
    σ : Type u_1
    τ : Type u_2
    R : Type u_4
    inst✝ : CommSemiring R
    f : σ → τ
    hf : Function.Injective f
    this : Eq (⇑(MvPolynomial.rename f)) (Finsupp.mapDomain (Finsupp.mapDomain f))
    ⊢ Function.Injective (Finsupp.mapDomain (Finsupp.mapDomain f))
  -/
  exact Finsupp.mapDomain_injective (Finsupp.mapDomain_injective hf)
  /-
    🎉 no goals
  -/


open Classical in
/-- Given a function between sets of variables `f : σ → τ` that is injective with proof `hf`,
  `MvPolynomial.killCompl hf` is the `AlgHom` from `R[τ]` to `R[σ]` that is left inverse to
  `rename f : R[σ] → R[τ]` and sends the variables in the complement of the range of `f` to `0`. -/
def killCompl : MvPolynomial τ R →ₐ[R] MvPolynomial σ R :=
  aeval fun i => if h : i ∈ Set.range f then X <| (Equiv.ofInjective f hf).symm ⟨i, h⟩ else 0


theorem killCompl_C (r : R) : killCompl hf (C r) = C r := algHom_C _ _


theorem killCompl_comp_rename : (killCompl hf).comp (rename f) = AlgHom.id R _ :=
  algHom_ext fun i => by
    /-
      σ : Type u_1
      τ : Type u_2
      R : Type u_4
      inst✝ : CommSemiring R
      f : σ → τ
      hf : Function.Injective f
      i : σ
      ⊢ Eq (((MvPolynomial.killCompl hf).comp (MvPolynomial.rename f)) (MvPolynomial …
    -/
    dsimp
    /-
      σ : Type u_1
      τ : Type u_2
      R : Type u_4
      inst✝ : CommSemiring R
      f : σ → τ
      hf : Function.Injective f
      i : σ
      ⊢ Eq ((MvPolynomial.killCompl hf) ((MvPolynomial.rename f) (MvPolynomial.X i)) …
    -/
    rw [rename, killCompl, aeval_X, comp_apply, aeval_X, dif_pos, Equiv.ofInjective_symm_apply]
    /-
      🎉 no goals
    -/


@[simp]
theorem killCompl_rename_app (p : MvPolynomial σ R) : killCompl hf (rename f p) = p :=
  AlgHom.congr_fun (killCompl_comp_rename hf) p


/-- `MvPolynomial.rename e` is an equivalence when `e` is. -/
@[simps apply]
def renameEquiv (f : σ ≃ τ) : MvPolynomial σ R ≃ₐ[R] MvPolynomial τ R :=
  { rename f with
    toFun := rename f
    invFun := rename f.symm
                            /-
                              σ : Type u_1
                              τ : Type u_2
                              α : Type u_3
                              R : Type u_4
                              S : Type u_5
                              inst✝¹ : CommSemiring R
                              inst✝ : CommSemiring S
                              f : Equiv σ τ
                              p : MvPolynomial σ R
                              ⊢ Eq ((MvPolynomial.rename ⇑f.symm) ((MvPolynomial.rename ⇑f) p)) p
                            -/
    left_inv := fun p => by rw [rename_rename, f.symm_comp_self, rename_id]
                            /-
                              🎉 no goals
                            -/
                             /-
                               σ : Type u_1
                               τ : Type u_2
                               α : Type u_3
                               R : Type u_4
                               S : Type u_5
                               inst✝¹ : CommSemiring R
                               inst✝ : CommSemiring S
                               f : Equiv σ τ
                               p : MvPolynomial τ R
                               ⊢ Eq ((MvPolynomial.rename ⇑f) ((MvPolynomial.rename ⇑f.symm) p)) p
                             -/
    right_inv := fun p => by rw [rename_rename, f.self_comp_symm, rename_id] }
                             /-
                               🎉 no goals
                             -/


@[simp]
theorem renameEquiv_refl : renameEquiv R (Equiv.refl σ) = AlgEquiv.refl :=
  AlgEquiv.ext rename_id


@[simp]
theorem renameEquiv_symm (f : σ ≃ τ) : (renameEquiv R f).symm = renameEquiv R f.symm :=
  rfl


@[simp]
theorem renameEquiv_trans (e : σ ≃ τ) (f : τ ≃ α) :
    (renameEquiv R e).trans (renameEquiv R f) = renameEquiv R (e.trans f) :=
  AlgEquiv.ext (rename_rename e f)


theorem eval₂_rename : (rename k p).eval₂ f g = p.eval₂ f (g ∘ k) := by
  /-
    σ : Type u_1
    τ : Type u_2
    R : Type u_4
    S : Type u_5
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S
    f : RingHom R S
    k : σ → τ
    g : τ → S
    p : MvPolynomial σ R
    ⊢ Eq (MvPolynomial.eval₂ f g ((MvPolynomial.rename k) p)) (MvPolynomial.eval₂  …
  -/
  apply MvPolynomial.induction_on p <;>
      /-
        case h_C
        σ : Type u_1
        τ : Type u_2
        R : Type u_4
        S : Type u_5
        inst✝¹ : CommSemiring R
        inst✝ : CommSemiring S
        f : RingHom R S
        k : σ → τ
        g : τ → S
        p : MvPolynomial σ R
        ⊢ ∀ (a : R), Eq (MvPolynomial.eval₂ f g ((MvPolynomial.rename k) (MvPolynomial …
      -/
      /-
        case h_C
        σ : Type u_1
        τ : Type u_2
        R : Type u_4
        S : Type u_5
        inst✝¹ : CommSemiring R
        inst✝ : CommSemiring S
        f : RingHom R S
        k : σ → τ
        g : τ → S
        p : MvPolynomial σ R
        a✝ : R
        ⊢ Eq (MvPolynomial.eval₂ f g ((MvPolynomial.rename k) (MvPolynomial.C a✝))) (M …
      -/
      /-
        🎉 no goals
      -/
      /-
        🎉 no goals
      -/
      /-
        case h_X
        σ : Type u_1
        τ : Type u_2
        R : Type u_4
        S : Type u_5
        inst✝¹ : CommSemiring R
        inst✝ : CommSemiring S
        f : RingHom R S
        k : σ → τ
        g : τ → S
        p p✝ : MvPolynomial σ R
        n✝ : σ
        a✝ : Eq (MvPolynomial.eval₂ f g ((MvPolynomial.rename k) p✝)) (MvPolynomial.ev …
        ⊢ Eq (MvPolynomial.eval₂ f g ((MvPolynomial.rename k) (HMul.hMul p✝ (MvPolynom …
      -/
      simp [*]
      /-
        🎉 no goals
      -/


theorem eval_rename (g : τ → R) (p : MvPolynomial σ R) : eval g (rename k p) = eval (g ∘ k) p :=
  eval₂_rename _ _ _ _


theorem eval₂Hom_rename : eval₂Hom f g (rename k p) = eval₂Hom f (g ∘ k) p :=
  eval₂_rename _ _ _ _


theorem aeval_rename [Algebra R S] : aeval g (rename k p) = aeval (g ∘ k) p :=
  eval₂Hom_rename _ _ _ _


theorem rename_eval₂ (g : τ → MvPolynomial σ R) :
    rename k (p.eval₂ C (g ∘ k)) = (rename k p).eval₂ C (rename k ∘ g) := by
  /-
    σ : Type u_1
    τ : Type u_2
    R : Type u_4
    inst✝ : CommSemiring R
    k : σ → τ
    p : MvPolynomial σ R
    g : τ → MvPolynomial σ R
    ⊢ Eq ((MvPolynomial.rename k) (MvPolynomial.eval₂ MvPolynomial.C (Function.com …
  -/
  apply MvPolynomial.induction_on p <;>
      /-
        case h_C
        σ : Type u_1
        τ : Type u_2
        R : Type u_4
        inst✝ : CommSemiring R
        k : σ → τ
        p : MvPolynomial σ R
        g : τ → MvPolynomial σ R
        ⊢ ∀ (a : R), Eq ((MvPolynomial.rename k) (MvPolynomial.eval₂ MvPolynomial.C (F …
      -/
      /-
        case h_C
        σ : Type u_1
        τ : Type u_2
        R : Type u_4
        inst✝ : CommSemiring R
        k : σ → τ
        p : MvPolynomial σ R
        g : τ → MvPolynomial σ R
        a✝ : R
        ⊢ Eq ((MvPolynomial.rename k) (MvPolynomial.eval₂ MvPolynomial.C (Function.com …
      -/
      /-
        🎉 no goals
      -/
      /-
        🎉 no goals
      -/
      /-
        case h_X
        σ : Type u_1
        τ : Type u_2
        R : Type u_4
        inst✝ : CommSemiring R
        k : σ → τ
        p : MvPolynomial σ R
        g : τ → MvPolynomial σ R
        p✝ : MvPolynomial σ R
        n✝ : σ
        a✝ : Eq ((MvPolynomial.rename k) (MvPolynomial.eval₂ MvPolynomial.C (Function. …
        ⊢ Eq ((MvPolynomial.rename k) (MvPolynomial.eval₂ MvPolynomial.C (Function.com …
      -/
      simp [*]
      /-
        🎉 no goals
      -/


theorem rename_prod_mk_eval₂ (j : τ) (g : σ → MvPolynomial σ R) :
    rename (Prod.mk j) (p.eval₂ C g) = p.eval₂ C fun x => rename (Prod.mk j) (g x) := by
  /-
    σ : Type u_1
    τ : Type u_2
    R : Type u_4
    inst✝ : CommSemiring R
    p : MvPolynomial σ R
    j : τ
    g : σ → MvPolynomial σ R
    ⊢ Eq ((MvPolynomial.rename (Prod.mk j)) (MvPolynomial.eval₂ MvPolynomial.C g p …
  -/
  apply MvPolynomial.induction_on p <;>
      /-
        case h_C
        σ : Type u_1
        τ : Type u_2
        R : Type u_4
        inst✝ : CommSemiring R
        p : MvPolynomial σ R
        j : τ
        g : σ → MvPolynomial σ R
        ⊢ ∀ (a : R), Eq ((MvPolynomial.rename (Prod.mk j)) (MvPolynomial.eval₂ MvPolyn …
      -/
      /-
        case h_C
        σ : Type u_1
        τ : Type u_2
        R : Type u_4
        inst✝ : CommSemiring R
        p : MvPolynomial σ R
        j : τ
        g : σ → MvPolynomial σ R
        a✝ : R
        ⊢ Eq ((MvPolynomial.rename (Prod.mk j)) (MvPolynomial.eval₂ MvPolynomial.C g ( …
      -/
      /-
        🎉 no goals
      -/
      /-
        🎉 no goals
      -/
      /-
        case h_X
        σ : Type u_1
        τ : Type u_2
        R : Type u_4
        inst✝ : CommSemiring R
        p : MvPolynomial σ R
        j : τ
        g : σ → MvPolynomial σ R
        p✝ : MvPolynomial σ R
        n✝ : σ
        a✝ : Eq ((MvPolynomial.rename (Prod.mk j)) (MvPolynomial.eval₂ MvPolynomial.C  …
        ⊢ Eq ((MvPolynomial.rename (Prod.mk j)) (MvPolynomial.eval₂ MvPolynomial.C g ( …
      -/
      simp [*]
      /-
        🎉 no goals
      -/


theorem eval₂_rename_prod_mk (g : σ × τ → S) (i : σ) (p : MvPolynomial τ R) :
    (rename (Prod.mk i) p).eval₂ f g = eval₂ f (fun j => g (i, j)) p := by
  /-
    σ : Type u_1
    τ : Type u_2
    R : Type u_4
    S : Type u_5
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S
    f : RingHom R S
    g : Prod σ τ → S
    i : σ
    p : MvPolynomial τ R
    ⊢ Eq (MvPolynomial.eval₂ f g ((MvPolynomial.rename (Prod.mk i)) p)) (MvPolynom …
  -/
  apply MvPolynomial.induction_on p <;>
      /-
        case h_C
        σ : Type u_1
        τ : Type u_2
        R : Type u_4
        S : Type u_5
        inst✝¹ : CommSemiring R
        inst✝ : CommSemiring S
        f : RingHom R S
        g : Prod σ τ → S
        i : σ
        p : MvPolynomial τ R
        ⊢ ∀ (a : R), Eq (MvPolynomial.eval₂ f g ((MvPolynomial.rename (Prod.mk i)) (Mv …
      -/
      /-
        case h_C
        σ : Type u_1
        τ : Type u_2
        R : Type u_4
        S : Type u_5
        inst✝¹ : CommSemiring R
        inst✝ : CommSemiring S
        f : RingHom R S
        g : Prod σ τ → S
        i : σ
        p : MvPolynomial τ R
        a✝ : R
        ⊢ Eq (MvPolynomial.eval₂ f g ((MvPolynomial.rename (Prod.mk i)) (MvPolynomial. …
      -/
      /-
        🎉 no goals
      -/
      /-
        🎉 no goals
      -/
      /-
        case h_X
        σ : Type u_1
        τ : Type u_2
        R : Type u_4
        S : Type u_5
        inst✝¹ : CommSemiring R
        inst✝ : CommSemiring S
        f : RingHom R S
        g : Prod σ τ → S
        i : σ
        p p✝ : MvPolynomial τ R
        n✝ : τ
        a✝ : Eq (MvPolynomial.eval₂ f g ((MvPolynomial.rename (Prod.mk i)) p✝)) (MvPol …
        ⊢ Eq (MvPolynomial.eval₂ f g ((MvPolynomial.rename (Prod.mk i)) (HMul.hMul p✝  …
      -/
      simp [*]
      /-
        🎉 no goals
      -/


theorem eval_rename_prod_mk (g : σ × τ → R) (i : σ) (p : MvPolynomial τ R) :
    eval g (rename (Prod.mk i) p) = eval (fun j => g (i, j)) p :=
  eval₂_rename_prod_mk (RingHom.id _) _ _ _


/-- Every polynomial is a polynomial in finitely many variables. -/
theorem exists_finset_rename (p : MvPolynomial σ R) :
    ∃ (s : Finset σ) (q : MvPolynomial { x // x ∈ s } R), p = rename (↑) q := by
  classical
  apply induction_on p
  · intro r
    exact ⟨∅, C r, by rw [rename_C]⟩
  · rintro p q ⟨s, p, rfl⟩ ⟨t, q, rfl⟩
    refine ⟨s ∪ t, ⟨?_, ?_⟩⟩
    · refine rename (Subtype.map id ?_) p + rename (Subtype.map id ?_) q <;>
        simp +contextual only [id, true_or, or_true,
          Finset.mem_union, forall_true_iff]
    · simp only [rename_rename, map_add]
      rfl
  · rintro p n ⟨s, p, rfl⟩
    refine ⟨insert n s, ⟨?_, ?_⟩⟩
    · refine rename (Subtype.map id ?_) p * X ⟨n, s.mem_insert_self n⟩
      simp +contextual only [id, or_true, Finset.mem_insert, forall_true_iff]
    · simp only [rename_rename, rename_X, Subtype.coe_mk, map_mul]
      rfl


/-- `exists_finset_rename` for two polynomials at once: for any two polynomials `p₁`, `p₂` in a
  polynomial semiring `R[σ]` of possibly infinitely many variables, `exists_finset_rename₂` yields
  a finite subset `s` of `σ` such that both `p₁` and `p₂` are contained in the polynomial semiring
  `R[s]` of finitely many variables. -/
theorem exists_finset_rename₂ (p₁ p₂ : MvPolynomial σ R) :
    ∃ (s : Finset σ) (q₁ q₂ : MvPolynomial s R), p₁ = rename (↑) q₁ ∧ p₂ = rename (↑) q₂ := by
  /-
    σ : Type u_1
    R : Type u_4
    inst✝ : CommSemiring R
    p₁ p₂ : MvPolynomial σ R
    ⊢ Exists fun s => Exists fun q₁ => Exists fun q₂ => And (Eq p₁ ((MvPolynomial. …
  -/
  obtain ⟨s₁, q₁, rfl⟩ := exists_finset_rename p₁
  /-
    case intro.intro
    σ : Type u_1
    R : Type u_4
    inst✝ : CommSemiring R
    p₂ : MvPolynomial σ R
    s₁ : Finset σ
    q₁ : MvPolynomial (Subtype fun x => Membership.mem s₁ x) R
    ⊢ Exists fun s => Exists fun q₁_1 => Exists fun q₂ => And (Eq ((MvPolynomial.r …
  -/
  obtain ⟨s₂, q₂, rfl⟩ := exists_finset_rename p₂
  classical
    use s₁ ∪ s₂
    use rename (Set.inclusion s₁.subset_union_left) q₁
    use rename (Set.inclusion s₁.subset_union_right) q₂
    constructor -- Porting note: was `<;> simp <;> rfl` but Lean couldn't infer the arguments
    · -- This used to be `rw`, but we need `erw` after https://github.com/leanprover/lean4/pull/2644
      erw [rename_rename (Set.inclusion s₁.subset_union_left)]
      rfl
    · -- This used to be `rw`, but we need `erw` after https://github.com/leanprover/lean4/pull/2644
      erw [rename_rename (Set.inclusion s₁.subset_union_right)]
      rfl


/-- Every polynomial is a polynomial in finitely many variables. -/
theorem exists_fin_rename (p : MvPolynomial σ R) :
    ∃ (n : ℕ) (f : Fin n → σ) (_hf : Injective f) (q : MvPolynomial (Fin n) R), p = rename f q := by
  /-
    σ : Type u_1
    R : Type u_4
    inst✝ : CommSemiring R
    p : MvPolynomial σ R
    ⊢ Exists fun n => Exists fun f => Exists fun _hf => Exists fun q => Eq p ((MvP …
  -/
  obtain ⟨s, q, rfl⟩ := exists_finset_rename p
  /-
    case intro.intro
    σ : Type u_1
    R : Type u_4
    inst✝ : CommSemiring R
    s : Finset σ
    q : MvPolynomial (Subtype fun x => Membership.mem s x) R
    ⊢ Exists fun n => Exists fun f => Exists fun _hf => Exists fun q_1 => Eq ((MvP …
  -/
  let n := Fintype.card { x // x ∈ s }
  /-
    case intro.intro
    σ : Type u_1
    R : Type u_4
    inst✝ : CommSemiring R
    s : Finset σ
    q : MvPolynomial (Subtype fun x => Membership.mem s x) R
    n : Nat := Fintype.card (Subtype fun x => Membership.mem s x)
    ⊢ Exists fun n => Exists fun f => Exists fun _hf => Exists fun q_1 => Eq ((MvP …
  -/
  let e := Fintype.equivFin { x // x ∈ s }
  /-
    case intro.intro
    σ : Type u_1
    R : Type u_4
    inst✝ : CommSemiring R
    s : Finset σ
    q : MvPolynomial (Subtype fun x => Membership.mem s x) R
    n : Nat := Fintype.card (Subtype fun x => Membership.mem s x)
    e : Equiv (Subtype fun x => Membership.mem s x) (Fin (Fintype.card (Subtype fu …
    ⊢ Exists fun n => Exists fun f => Exists fun _hf => Exists fun q_1 => Eq ((MvP …
  -/
  refine ⟨n, (↑) ∘ e.symm, Subtype.val_injective.comp e.symm.injective, rename e q, ?_⟩
  /-
    case intro.intro
    σ : Type u_1
    R : Type u_4
    inst✝ : CommSemiring R
    s : Finset σ
    q : MvPolynomial (Subtype fun x => Membership.mem s x) R
    n : Nat := Fintype.card (Subtype fun x => Membership.mem s x)
    e : Equiv (Subtype fun x => Membership.mem s x) (Fin (Fintype.card (Subtype fu …
    ⊢ Eq ((MvPolynomial.rename Subtype.val) q) ((MvPolynomial.rename (Function.com …
  -/
  rw [← rename_rename, rename_rename e]
  /-
    case intro.intro
    σ : Type u_1
    R : Type u_4
    inst✝ : CommSemiring R
    s : Finset σ
    q : MvPolynomial (Subtype fun x => Membership.mem s x) R
    n : Nat := Fintype.card (Subtype fun x => Membership.mem s x)
    e : Equiv (Subtype fun x => Membership.mem s x) (Fin (Fintype.card (Subtype fu …
    ⊢ Eq ((MvPolynomial.rename Subtype.val) q) ((MvPolynomial.rename Subtype.val)  …
  -/
  simp only [Function.comp_def, Equiv.symm_apply_apply, rename_rename]
  /-
    🎉 no goals
  -/


theorem eval₂_cast_comp (f : σ → τ) (c : ℤ →+* R) (g : τ → R) (p : MvPolynomial σ ℤ) :
    eval₂ c (g ∘ f) p = eval₂ c g (rename f p) := by
  apply MvPolynomial.induction_on p (fun n => by simp only [eval₂_C, rename_C])
    (fun p q hp hq => by simp only [hp, hq, rename, eval₂_add, map_add])
    fun p n hp => by simp only [eval₂_mul, hp, eval₂_X, comp_apply, map_mul, rename_X, eval₂_mul]


@[simp]
theorem coeff_rename_mapDomain (f : σ → τ) (hf : Injective f) (φ : MvPolynomial σ R) (d : σ →₀ ℕ) :
    (rename f φ).coeff (d.mapDomain f) = φ.coeff d := by
  classical
  apply φ.induction_on' (P := fun ψ => coeff (Finsupp.mapDomain f d) ((rename f) ψ) = coeff d ψ)
  -- Lean could no longer infer the motive
  · intro u r
    rw [rename_monomial, coeff_monomial, coeff_monomial]
    simp only [(Finsupp.mapDomain_injective hf).eq_iff]
  · intros
    simp only [*, map_add, coeff_add]


@[simp]
theorem coeff_rename_embDomain (f : σ ↪ τ) (φ : MvPolynomial σ R) (d : σ →₀ ℕ) :
    (rename f φ).coeff (d.embDomain f) = φ.coeff d := by
  /-
    σ : Type u_1
    τ : Type u_2
    R : Type u_4
    inst✝ : CommSemiring R
    f : Function.Embedding σ τ
    φ : MvPolynomial σ R
    d : Finsupp σ Nat
    ⊢ Eq (MvPolynomial.coeff (Finsupp.embDomain f d) ((MvPolynomial.rename ⇑f) φ)) …
  -/
  rw [Finsupp.embDomain_eq_mapDomain f, coeff_rename_mapDomain f f.injective]
  /-
    🎉 no goals
  -/


theorem coeff_rename_eq_zero (f : σ → τ) (φ : MvPolynomial σ R) (d : τ →₀ ℕ)
    (h : ∀ u : σ →₀ ℕ, u.mapDomain f = d → φ.coeff u = 0) : (rename f φ).coeff d = 0 := by
  classical
  rw [rename_eq, ← not_mem_support_iff]
  intro H
  replace H := mapDomain_support H
  rw [Finset.mem_image] at H
  obtain ⟨u, hu, rfl⟩ := H
  specialize h u rfl
  simp? at h hu says simp only [Finsupp.mem_support_iff, ne_eq] at h hu
  contradiction


theorem coeff_rename_ne_zero (f : σ → τ) (φ : MvPolynomial σ R) (d : τ →₀ ℕ)
    (h : (rename f φ).coeff d ≠ 0) : ∃ u : σ →₀ ℕ, u.mapDomain f = d ∧ φ.coeff u ≠ 0 := by
  /-
    σ : Type u_1
    τ : Type u_2
    R : Type u_4
    inst✝ : CommSemiring R
    f : σ → τ
    φ : MvPolynomial σ R
    d : Finsupp τ Nat
    h : Ne (MvPolynomial.coeff d ((MvPolynomial.rename f) φ)) 0
    ⊢ Exists fun u => And (Eq (Finsupp.mapDomain f u) d) (Ne (MvPolynomial.coeff u …
  -/
  contrapose! h
  /-
    σ : Type u_1
    τ : Type u_2
    R : Type u_4
    inst✝ : CommSemiring R
    f : σ → τ
    φ : MvPolynomial σ R
    d : Finsupp τ Nat
    h : ∀ (u : Finsupp σ Nat), Eq (Finsupp.mapDomain f u) d → Eq (MvPolynomial.coe …
    ⊢ Eq (MvPolynomial.coeff d ((MvPolynomial.rename f) φ)) 0
  -/
  apply coeff_rename_eq_zero _ _ _ h
  /-
    🎉 no goals
  -/


@[simp]
theorem constantCoeff_rename {τ : Type*} (f : σ → τ) (φ : MvPolynomial σ R) :
    constantCoeff (rename f φ) = constantCoeff φ := by
  /-
    σ : Type u_1
    R : Type u_4
    inst✝ : CommSemiring R
    τ : Type u_6
    f : σ → τ
    φ : MvPolynomial σ R
    ⊢ Eq (MvPolynomial.constantCoeff ((MvPolynomial.rename f) φ)) (MvPolynomial.co …
  -/
  apply φ.induction_on
    /-
      case h_C
      σ : Type u_1
      R : Type u_4
      inst✝ : CommSemiring R
      τ : Type u_6
      f : σ → τ
      φ : MvPolynomial σ R
      ⊢ ∀ (a : R), Eq (MvPolynomial.constantCoeff ((MvPolynomial.rename f) (MvPolyno …
    -/
  · intro a
    /-
      case h_C
      σ : Type u_1
      R : Type u_4
      inst✝ : CommSemiring R
      τ : Type u_6
      f : σ → τ
      φ : MvPolynomial σ R
      a : R
      ⊢ Eq (MvPolynomial.constantCoeff ((MvPolynomial.rename f) (MvPolynomial.C a))) …
    -/
    simp only [constantCoeff_C, rename_C]
    /-
      🎉 no goals
    -/
    /-
      case h_add
      σ : Type u_1
      R : Type u_4
      inst✝ : CommSemiring R
      τ : Type u_6
      f : σ → τ
      φ : MvPolynomial σ R
      ⊢ ∀ (p q : MvPolynomial σ R), Eq (MvPolynomial.constantCoeff ((MvPolynomial.re …
    -/
  · intro p q hp hq
    /-
      case h_add
      σ : Type u_1
      R : Type u_4
      inst✝ : CommSemiring R
      τ : Type u_6
      f : σ → τ
      φ p q : MvPolynomial σ R
      hp : Eq (MvPolynomial.constantCoeff ((MvPolynomial.rename f) p)) (MvPolynomial …
      hq : Eq (MvPolynomial.constantCoeff ((MvPolynomial.rename f) q)) (MvPolynomial …
      ⊢ Eq (MvPolynomial.constantCoeff ((MvPolynomial.rename f) (HAdd.hAdd p q))) (M …
    -/
    simp only [hp, hq, map_add]
    /-
      🎉 no goals
    -/
    /-
      case h_X
      σ : Type u_1
      R : Type u_4
      inst✝ : CommSemiring R
      τ : Type u_6
      f : σ → τ
      φ : MvPolynomial σ R
      ⊢ ∀ (p : MvPolynomial σ R) (n : σ), Eq (MvPolynomial.constantCoeff ((MvPolynom …
    -/
  · intro p n hp
    /-
      case h_X
      σ : Type u_1
      R : Type u_4
      inst✝ : CommSemiring R
      τ : Type u_6
      f : σ → τ
      φ p : MvPolynomial σ R
      n : σ
      hp : Eq (MvPolynomial.constantCoeff ((MvPolynomial.rename f) p)) (MvPolynomial …
      ⊢ Eq (MvPolynomial.constantCoeff ((MvPolynomial.rename f) (HMul.hMul p (MvPoly …
    -/
    simp only [hp, rename_X, constantCoeff_X, map_mul]
    /-
      🎉 no goals
    -/


theorem support_rename_of_injective {p : MvPolynomial σ R} {f : σ → τ} [DecidableEq τ]
    (h : Function.Injective f) :
    (rename f p).support = Finset.image (Finsupp.mapDomain f) p.support := by
  /-
    σ : Type u_1
    τ : Type u_2
    R : Type u_4
    inst✝¹ : CommSemiring R
    p : MvPolynomial σ R
    f : σ → τ
    inst✝ : DecidableEq τ
    h : Function.Injective f
    ⊢ Eq ((MvPolynomial.rename f) p).support (Finset.image (Finsupp.mapDomain f) p …
  -/
  rw [rename_eq]
  /-
    σ : Type u_1
    τ : Type u_2
    R : Type u_4
    inst✝¹ : CommSemiring R
    p : MvPolynomial σ R
    f : σ → τ
    inst✝ : DecidableEq τ
    h : Function.Injective f
    ⊢ Eq (MvPolynomial.support (Finsupp.mapDomain (Finsupp.mapDomain f) p)) (Finse …
  -/
  exact Finsupp.mapDomain_support_of_injective (mapDomain_injective h) _
  /-
    🎉 no goals
  -/


