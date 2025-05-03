/-- The quotient of a Lie module by a Lie submodule. It is a Lie module. -/
instance : HasQuotient M (LieSubmodule R L M) :=
  ⟨fun N => M ⧸ N.toSubmodule⟩


instance addCommGroup : AddCommGroup (M ⧸ N) :=
  Submodule.Quotient.addCommGroup _


instance module' {S : Type*} [Semiring S] [SMul S R] [Module S M] [IsScalarTower S R M] :
    Module S (M ⧸ N) :=
  Submodule.Quotient.module' _


instance module : Module R (M ⧸ N) :=
  Submodule.Quotient.module _


instance isCentralScalar {S : Type*} [Semiring S] [SMul S R] [Module S M] [IsScalarTower S R M]
    [SMul Sᵐᵒᵖ R] [Module Sᵐᵒᵖ M] [IsScalarTower Sᵐᵒᵖ R M] [IsCentralScalar S M] :
    IsCentralScalar S (M ⧸ N) :=
  Submodule.Quotient.isCentralScalar _


instance inhabited : Inhabited (M ⧸ N) :=
  ⟨0⟩


/-- Map sending an element of `M` to the corresponding element of `M/N`, when `N` is a
lie_submodule of the lie_module `N`. -/
abbrev mk : M → M ⧸ N :=
  Submodule.Quotient.mk

-- Porting note: added to replace `mk_eq_zero` as simp lemma.

@[simp]
theorem mk_eq_zero' {m : M} : mk (N := N) m = 0 ↔ m ∈ N :=
  Submodule.Quotient.mk_eq_zero N.toSubmodule


theorem is_quotient_mk (m : M) : Quotient.mk'' m = (mk m : M ⧸ N) :=
  rfl


/-- Given a Lie module `M` over a Lie algebra `L`, together with a Lie submodule `N ⊆ M`, there
is a natural linear map from `L` to the endomorphisms of `M` leaving `N` invariant. -/
def lieSubmoduleInvariant : L →ₗ[R] Submodule.compatibleMaps N.toSubmodule N.toSubmodule :=
  LinearMap.codRestrict _ (LieModule.toEnd R L M) fun _ _ => N.lie_mem


/-- Given a Lie module `M` over a Lie algebra `L`, together with a Lie submodule `N ⊆ M`, there
is a natural Lie algebra morphism from `L` to the linear endomorphism of the quotient `M/N`. -/
def actionAsEndoMap : L →ₗ⁅R⁆ Module.End R (M ⧸ N) :=
  { LinearMap.comp (Submodule.mapQLinear (N : Submodule R M) (N : Submodule R M))
      lieSubmoduleInvariant with
    map_lie' := fun {_ _} =>
      Submodule.linearMap_qext _ <| LinearMap.ext fun _ => congr_arg mk <| lie_lie _ _ _ }


/-- Given a Lie module `M` over a Lie algebra `L`, together with a Lie submodule `N ⊆ M`, there is
a natural bracket action of `L` on the quotient `M/N`. -/
instance actionAsEndoMapBracket : Bracket L (M ⧸ N) :=
  ⟨fun x n => actionAsEndoMap N x n⟩


instance lieQuotientLieRingModule : LieRingModule L (M ⧸ N) :=
  { LieRingModule.compLieHom _ (actionAsEndoMap N) with bracket := Bracket.bracket }


/-- The quotient of a Lie module by a Lie submodule, is a Lie module. -/
instance lieQuotientLieModule : LieModule R L (M ⧸ N) :=
  LieModule.compLieHom _ (actionAsEndoMap N)


instance lieQuotientHasBracket : Bracket (L ⧸ I) (L ⧸ I) :=
  ⟨by
    /-
      R : Type u
      L : Type v
      M : Type w
      inst✝⁶ : CommRing R
      inst✝⁵ : LieRing L
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : LieRingModule L M
      N N' : LieSubmodule R L M
      inst✝¹ : LieAlgebra R L
      inst✝ : LieModule R L M
      I J : LieIdeal R L
      ⊢ HasQuotient.Quotient L I → HasQuotient.Quotient L I → HasQuotient.Quotient L I
    -/
    intro x y
    /-
      R : Type u
      L : Type v
      M : Type w
      inst✝⁶ : CommRing R
      inst✝⁵ : LieRing L
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : LieRingModule L M
      N N' : LieSubmodule R L M
      inst✝¹ : LieAlgebra R L
      inst✝ : LieModule R L M
      I J : LieIdeal R L
      x y : HasQuotient.Quotient L I
      ⊢ HasQuotient.Quotient L I
    -/
    apply Quotient.liftOn₂' x y fun x' y' => mk ⁅x', y'⁆
    /-
      R : Type u
      L : Type v
      M : Type w
      inst✝⁶ : CommRing R
      inst✝⁵ : LieRing L
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : LieRingModule L M
      N N' : LieSubmodule R L M
      inst✝¹ : LieAlgebra R L
      inst✝ : LieModule R L M
      I J : LieIdeal R L
      x y : HasQuotient.Quotient L I
      ⊢ ∀ (a₁ a₂ b₁ b₂ : L), (↑I).quotientRel a₁ b₁ → (↑I).quotientRel a₂ b₂ → Eq (L …
    -/
    intro x₁ x₂ y₁ y₂ h₁ h₂
    /-
      R : Type u
      L : Type v
      M : Type w
      inst✝⁶ : CommRing R
      inst✝⁵ : LieRing L
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : LieRingModule L M
      N N' : LieSubmodule R L M
      inst✝¹ : LieAlgebra R L
      inst✝ : LieModule R L M
      I J : LieIdeal R L
      x y : HasQuotient.Quotient L I
      x₁ x₂ y₁ y₂ : L
      h₁ : (↑I).quotientRel x₁ y₁
      h₂ : (↑I).quotientRel x₂ y₂
      ⊢ Eq (LieSubmodule.Quotient.mk (Bracket.bracket x₁ x₂)) (LieSubmodule.Quotient …
    -/
    apply (Submodule.Quotient.eq I.toSubmodule).2
    /-
      R : Type u
      L : Type v
      M : Type w
      inst✝⁶ : CommRing R
      inst✝⁵ : LieRing L
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : LieRingModule L M
      N N' : LieSubmodule R L M
      inst✝¹ : LieAlgebra R L
      inst✝ : LieModule R L M
      I J : LieIdeal R L
      x y : HasQuotient.Quotient L I
      x₁ x₂ y₁ y₂ : L
      h₁ : (↑I).quotientRel x₁ y₁
      h₂ : (↑I).quotientRel x₂ y₂
      ⊢ Membership.mem (↑I) (HSub.hSub (Bracket.bracket x₁ x₂) (Bracket.bracket y₁ y …
    -/
    rw [Submodule.quotientRel_def] at h₁ h₂
    have h : ⁅x₁, x₂⁆ - ⁅y₁, y₂⁆ = ⁅x₁, x₂ - y₂⁆ + ⁅x₁ - y₁, y₂⁆ := by
      simp [-lie_skew, sub_eq_add_neg, add_assoc]
    /-
      R : Type u
      L : Type v
      M : Type w
      inst✝⁶ : CommRing R
      inst✝⁵ : LieRing L
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : LieRingModule L M
      N N' : LieSubmodule R L M
      inst✝¹ : LieAlgebra R L
      inst✝ : LieModule R L M
      I J : LieIdeal R L
      x y : HasQuotient.Quotient L I
      x₁ x₂ y₁ y₂ : L
      h₁ : Membership.mem (↑I) (HSub.hSub x₁ y₁)
      h₂ : Membership.mem (↑I) (HSub.hSub x₂ y₂)
      h : Eq (HSub.hSub (Bracket.bracket x₁ x₂) (Bracket.bracket y₁ y₂)) (HAdd.hAdd  …
      ⊢ Membership.mem (↑I) (HSub.hSub (Bracket.bracket x₁ x₂) (Bracket.bracket y₁ y …
    -/
    rw [h]
    /-
      R : Type u
      L : Type v
      M : Type w
      inst✝⁶ : CommRing R
      inst✝⁵ : LieRing L
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : LieRingModule L M
      N N' : LieSubmodule R L M
      inst✝¹ : LieAlgebra R L
      inst✝ : LieModule R L M
      I J : LieIdeal R L
      x y : HasQuotient.Quotient L I
      x₁ x₂ y₁ y₂ : L
      h₁ : Membership.mem (↑I) (HSub.hSub x₁ y₁)
      h₂ : Membership.mem (↑I) (HSub.hSub x₂ y₂)
      h : Eq (HSub.hSub (Bracket.bracket x₁ x₂) (Bracket.bracket y₁ y₂)) (HAdd.hAdd  …
      ⊢ Membership.mem (↑I) (HAdd.hAdd (Bracket.bracket x₁ (HSub.hSub x₂ y₂)) (Brack …
    -/
    apply Submodule.add_mem
      /-
        case h₁
        R : Type u
        L : Type v
        M : Type w
        inst✝⁶ : CommRing R
        inst✝⁵ : LieRing L
        inst✝⁴ : AddCommGroup M
        inst✝³ : Module R M
        inst✝² : LieRingModule L M
        N N' : LieSubmodule R L M
        inst✝¹ : LieAlgebra R L
        inst✝ : LieModule R L M
        I J : LieIdeal R L
        x y : HasQuotient.Quotient L I
        x₁ x₂ y₁ y₂ : L
        h₁ : Membership.mem (↑I) (HSub.hSub x₁ y₁)
        h₂ : Membership.mem (↑I) (HSub.hSub x₂ y₂)
        h : Eq (HSub.hSub (Bracket.bracket x₁ x₂) (Bracket.bracket y₁ y₂)) (HAdd.hAdd  …
        ⊢ Membership.mem (↑I) (Bracket.bracket x₁ (HSub.hSub x₂ y₂))
      -/
    · apply lie_mem_right R L I x₁ (x₂ - y₂) h₂
      /-
        🎉 no goals
      -/
      /-
        case h₂
        R : Type u
        L : Type v
        M : Type w
        inst✝⁶ : CommRing R
        inst✝⁵ : LieRing L
        inst✝⁴ : AddCommGroup M
        inst✝³ : Module R M
        inst✝² : LieRingModule L M
        N N' : LieSubmodule R L M
        inst✝¹ : LieAlgebra R L
        inst✝ : LieModule R L M
        I J : LieIdeal R L
        x y : HasQuotient.Quotient L I
        x₁ x₂ y₁ y₂ : L
        h₁ : Membership.mem (↑I) (HSub.hSub x₁ y₁)
        h₂ : Membership.mem (↑I) (HSub.hSub x₂ y₂)
        h : Eq (HSub.hSub (Bracket.bracket x₁ x₂) (Bracket.bracket y₁ y₂)) (HAdd.hAdd  …
        ⊢ Membership.mem (↑I) (Bracket.bracket (HSub.hSub x₁ y₁) y₂)
      -/
    · apply lie_mem_left R L I (x₁ - y₁) y₂ h₁⟩
      /-
        🎉 no goals
      -/


@[simp]
theorem mk_bracket (x y : L) : mk ⁅x, y⁆ = ⁅(mk x : L ⧸ I), (mk y : L ⧸ I)⁆ :=
  rfl


instance lieQuotientLieRing : LieRing (L ⧸ I) where
  add_lie := by
    /-
      R : Type u
      L : Type v
      M : Type w
      inst✝⁶ : CommRing R
      inst✝⁵ : LieRing L
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : LieRingModule L M
      N N' : LieSubmodule R L M
      inst✝¹ : LieAlgebra R L
      inst✝ : LieModule R L M
      I J : LieIdeal R L
      ⊢ ∀ (x y z : HasQuotient.Quotient L I), Eq (Bracket.bracket (HAdd.hAdd x y) z) …
    -/
    intro x' y' z'; refine Quotient.inductionOn₃' x' y' z' ?_; intro x y z
    repeat'
      first
      | rw [is_quotient_mk]
      | rw [← mk_bracket]
      | rw [← Submodule.Quotient.mk_add (R := R) (M := L)]
    /-
      R : Type u
      L : Type v
      M : Type w
      inst✝⁶ : CommRing R
      inst✝⁵ : LieRing L
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : LieRingModule L M
      N N' : LieSubmodule R L M
      inst✝¹ : LieAlgebra R L
      inst✝ : LieModule R L M
      I J : LieIdeal R L
      x' y' z' : HasQuotient.Quotient L I
      x y z : L
      ⊢ Eq (LieSubmodule.Quotient.mk (Bracket.bracket (HAdd.hAdd x y) z)) (Submodule …
    -/
    apply congr_arg; apply add_lie
                     /-
                       🎉 no goals
                     -/
  lie_add := by
    /-
      R : Type u
      L : Type v
      M : Type w
      inst✝⁶ : CommRing R
      inst✝⁵ : LieRing L
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : LieRingModule L M
      N N' : LieSubmodule R L M
      inst✝¹ : LieAlgebra R L
      inst✝ : LieModule R L M
      I J : LieIdeal R L
      ⊢ ∀ (x y z : HasQuotient.Quotient L I), Eq (Bracket.bracket x (HAdd.hAdd y z)) …
    -/
    intro x' y' z'; refine Quotient.inductionOn₃' x' y' z' ?_; intro x y z
    repeat'
      first
      | rw [is_quotient_mk]
      | rw [← mk_bracket]
      | rw [← Submodule.Quotient.mk_add (R := R) (M := L)]
    /-
      R : Type u
      L : Type v
      M : Type w
      inst✝⁶ : CommRing R
      inst✝⁵ : LieRing L
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : LieRingModule L M
      N N' : LieSubmodule R L M
      inst✝¹ : LieAlgebra R L
      inst✝ : LieModule R L M
      I J : LieIdeal R L
      x' y' z' : HasQuotient.Quotient L I
      x y z : L
      ⊢ Eq (LieSubmodule.Quotient.mk (Bracket.bracket x (HAdd.hAdd y z))) (Submodule …
    -/
    apply congr_arg; apply lie_add
                     /-
                       🎉 no goals
                     -/
  lie_self := by
    /-
      R : Type u
      L : Type v
      M : Type w
      inst✝⁶ : CommRing R
      inst✝⁵ : LieRing L
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : LieRingModule L M
      N N' : LieSubmodule R L M
      inst✝¹ : LieAlgebra R L
      inst✝ : LieModule R L M
      I J : LieIdeal R L
      ⊢ ∀ (x : HasQuotient.Quotient L I), Eq (Bracket.bracket x x) 0
    -/
    intro x'; refine Quotient.inductionOn' x' ?_; intro x
    /-
      R : Type u
      L : Type v
      M : Type w
      inst✝⁶ : CommRing R
      inst✝⁵ : LieRing L
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : LieRingModule L M
      N N' : LieSubmodule R L M
      inst✝¹ : LieAlgebra R L
      inst✝ : LieModule R L M
      I J : LieIdeal R L
      x' : HasQuotient.Quotient L I
      x : L
      ⊢ Eq (Bracket.bracket (Quotient.mk'' x) (Quotient.mk'' x)) 0
    -/
    rw [is_quotient_mk, ← mk_bracket]
    /-
      R : Type u
      L : Type v
      M : Type w
      inst✝⁶ : CommRing R
      inst✝⁵ : LieRing L
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : LieRingModule L M
      N N' : LieSubmodule R L M
      inst✝¹ : LieAlgebra R L
      inst✝ : LieModule R L M
      I J : LieIdeal R L
      x' : HasQuotient.Quotient L I
      x : L
      ⊢ Eq (LieSubmodule.Quotient.mk (Bracket.bracket x x)) 0
    -/
    apply congr_arg; apply lie_self
                     /-
                       🎉 no goals
                     -/
  leibniz_lie := by
    /-
      R : Type u
      L : Type v
      M : Type w
      inst✝⁶ : CommRing R
      inst✝⁵ : LieRing L
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : LieRingModule L M
      N N' : LieSubmodule R L M
      inst✝¹ : LieAlgebra R L
      inst✝ : LieModule R L M
      I J : LieIdeal R L
      ⊢ ∀ (x y z : HasQuotient.Quotient L I), Eq (Bracket.bracket x (Bracket.bracket …
    -/
    intro x' y' z'; refine Quotient.inductionOn₃' x' y' z' ?_; intro x y z
    repeat'
      first
      | rw [is_quotient_mk]
      | rw [← mk_bracket]
      | rw [← Submodule.Quotient.mk_add (R := R) (M := L)]
    /-
      R : Type u
      L : Type v
      M : Type w
      inst✝⁶ : CommRing R
      inst✝⁵ : LieRing L
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : LieRingModule L M
      N N' : LieSubmodule R L M
      inst✝¹ : LieAlgebra R L
      inst✝ : LieModule R L M
      I J : LieIdeal R L
      x' y' z' : HasQuotient.Quotient L I
      x y z : L
      ⊢ Eq (LieSubmodule.Quotient.mk (Bracket.bracket x (Bracket.bracket y z))) (Sub …
    -/
    apply congr_arg; apply leibniz_lie
                     /-
                       🎉 no goals
                     -/


instance lieQuotientLieAlgebra : LieAlgebra R (L ⧸ I) where
  lie_smul := by
    /-
      R : Type u
      L : Type v
      M : Type w
      inst✝⁶ : CommRing R
      inst✝⁵ : LieRing L
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : LieRingModule L M
      N N' : LieSubmodule R L M
      inst✝¹ : LieAlgebra R L
      inst✝ : LieModule R L M
      I J : LieIdeal R L
      ⊢ ∀ (t : R) (x y : HasQuotient.Quotient L I), Eq (Bracket.bracket x (HSMul.hSM …
    -/
    intro t x' y'; refine Quotient.inductionOn₂' x' y' ?_; intro x y
    repeat'
      first
      | rw [is_quotient_mk]
      | rw [← mk_bracket]
      | rw [← Submodule.Quotient.mk_smul (R := R) (M := L)]
    /-
      R : Type u
      L : Type v
      M : Type w
      inst✝⁶ : CommRing R
      inst✝⁵ : LieRing L
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : LieRingModule L M
      N N' : LieSubmodule R L M
      inst✝¹ : LieAlgebra R L
      inst✝ : LieModule R L M
      I J : LieIdeal R L
      t : R
      x' y' : HasQuotient.Quotient L I
      x y : L
      ⊢ Eq (LieSubmodule.Quotient.mk (Bracket.bracket x (HSMul.hSMul t y))) (Submodu …
    -/
    apply congr_arg; apply lie_smul
                     /-
                       🎉 no goals
                     -/


/-- `LieSubmodule.Quotient.mk` as a `LieModuleHom`. -/
@[simps]
def mk' : M →ₗ⁅R,L⁆ M ⧸ N :=
  { N.toSubmodule.mkQ with
    toFun := mk
    map_lie' := fun {_ _} => rfl }


@[simp]
theorem surjective_mk' : Function.Surjective (mk' N) := Quot.mk_surjective


@[simp]
theorem range_mk' : LieModuleHom.range (mk' N) = ⊤ := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : LieRingModule L M
    N : LieSubmodule R L M
    inst✝¹ : LieAlgebra R L
    inst✝ : LieModule R L M
    ⊢ Eq (LieSubmodule.Quotient.mk' N).range Top.top
  -/
  simp [LieModuleHom.range_eq_top]
  /-
    🎉 no goals
  -/


instance isNoetherian [IsNoetherian R M] : IsNoetherian R (M ⧸ N) :=
  inferInstanceAs (IsNoetherian R (M ⧸ (N : Submodule R M)))

-- Porting note: LHS simplifies @[simp]

theorem mk_eq_zero {m : M} : mk' N m = 0 ↔ m ∈ N :=
  Submodule.Quotient.mk_eq_zero N.toSubmodule


@[simp]
                                        /-
                                          R : Type u
                                          L : Type v
                                          M : Type w
                                          inst✝⁶ : CommRing R
                                          inst✝⁵ : LieRing L
                                          inst✝⁴ : AddCommGroup M
                                          inst✝³ : Module R M
                                          inst✝² : LieRingModule L M
                                          N : LieSubmodule R L M
                                          inst✝¹ : LieAlgebra R L
                                          inst✝ : LieModule R L M
                                          ⊢ Eq (LieSubmodule.Quotient.mk' N).ker N
                                        -/
theorem mk'_ker : (mk' N).ker = N := by ext; simp
                                             /-
                                               🎉 no goals
                                             -/


@[simp]
theorem map_mk'_eq_bot_le : map (mk' N) N' = ⊥ ↔ N' ≤ N := by
  /-
    R : Type u
    L : Type v
    M : Type w
    inst✝⁶ : CommRing R
    inst✝⁵ : LieRing L
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : LieRingModule L M
    N N' : LieSubmodule R L M
    inst✝¹ : LieAlgebra R L
    inst✝ : LieModule R L M
    ⊢ Iff (Eq (LieSubmodule.map (LieSubmodule.Quotient.mk' N) N') Bot.bot) (LE.le  …
  -/
  rw [← LieModuleHom.le_ker_iff_map, mk'_ker]
  /-
    🎉 no goals
  -/


/-- Two `LieModuleHom`s from a quotient lie module are equal if their compositions with
`LieSubmodule.Quotient.mk'` are equal.

See note [partially-applied ext lemmas]. -/
@[ext]
theorem lieModuleHom_ext ⦃f g : M ⧸ N →ₗ⁅R,L⁆ M⦄ (h : f.comp (mk' N) = g.comp (mk' N)) : f = g :=
  LieModuleHom.ext fun x => Quotient.inductionOn' x <| LieModuleHom.congr_fun h


lemma toEnd_comp_mk' (x : L) :
    LieModule.toEnd R L (M ⧸ N) x ∘ₗ mk' N = mk' N ∘ₗ LieModule.toEnd R L M x :=
  rfl


/-- The first isomorphism theorem for morphisms of Lie algebras. -/
@[simps]
noncomputable def quotKerEquivRange : (L ⧸ f.ker) ≃ₗ⁅R⁆ f.range :=
  { (f : L →ₗ[R] L').quotKerEquivRange with
    toFun := (f : L →ₗ[R] L').quotKerEquivRange
    map_lie' := by
      /-
        R : Type u_1
        L : Type u_2
        L' : Type u_3
        inst✝⁴ : CommRing R
        inst✝³ : LieRing L
        inst✝² : LieAlgebra R L
        inst✝¹ : LieRing L'
        inst✝ : LieAlgebra R L'
        f : LieHom R L L'
        ⊢ ∀ {x y : HasQuotient.Quotient L f.ker}, Eq ({ toFun := ⇑(↑f).quotKerEquivRan …
      -/
      rintro ⟨x⟩ ⟨y⟩
      /-
        case mk.mk
        R : Type u_1
        L : Type u_2
        L' : Type u_3
        inst✝⁴ : CommRing R
        inst✝³ : LieRing L
        inst✝² : LieAlgebra R L
        inst✝¹ : LieRing L'
        inst✝ : LieAlgebra R L'
        f : LieHom R L L'
        x✝ : HasQuotient.Quotient L f.ker
        x : L
        y✝ : HasQuotient.Quotient L f.ker
        y : L
        ⊢ Eq ({ toFun := ⇑(↑f).quotKerEquivRange, map_add' := ⋯, map_smul' := ⋯ }.toFu …
      -/
      rw [← SetLike.coe_eq_coe, LieSubalgebra.coe_bracket]
      simp only [Submodule.Quotient.quot_mk_eq_mk, LinearMap.quotKerEquivRange_apply_mk, ←
        LieSubmodule.Quotient.mk_bracket, coe_toLinearMap, map_lie] }


