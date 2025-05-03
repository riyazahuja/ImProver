/-- A decomposition is an equivalence between an additive monoid `M` and a direct sum of additive
submonoids `ℳ i` of that `M`, such that the "recomposition" is canonical. This definition also
works for additive groups and modules.

This is a version of `DirectSum.IsInternal` which comes with a constructive inverse to the
canonical "recomposition" rather than just a proof that the "recomposition" is bijective.

Often it is easier to construct a term of this type via `Decomposition.ofAddHom` or
`Decomposition.ofLinearMap`. -/
class Decomposition where
  decompose' : M → ⨁ i, ℳ i
  left_inv : Function.LeftInverse (DirectSum.coeAddMonoidHom ℳ) decompose'
  right_inv : Function.RightInverse (DirectSum.coeAddMonoidHom ℳ) decompose'


/-- `DirectSum.Decomposition` instances, while carrying data, are always equal. -/
instance : Subsingleton (Decomposition ℳ) :=
  ⟨fun x y ↦ by
    /-
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      σ : Type u_4
      inst✝³ : DecidableEq ι
      inst✝² : AddCommMonoid M
      inst✝¹ : SetLike σ M
      inst✝ : AddSubmonoidClass σ M
      ℳ : ι → σ
      x y : DirectSum.Decomposition ℳ
      ⊢ Eq x y
    -/
    obtain ⟨_, _, xr⟩ := x
    /-
      case mk
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      σ : Type u_4
      inst✝³ : DecidableEq ι
      inst✝² : AddCommMonoid M
      inst✝¹ : SetLike σ M
      inst✝ : AddSubmonoidClass σ M
      ℳ : ι → σ
      y : DirectSum.Decomposition ℳ
      decompose'✝ : M → DirectSum ι fun i => Subtype fun x => Membership.mem (ℳ i) x
      left_inv✝ : Function.LeftInverse (⇑(DirectSum.coeAddMonoidHom ℳ)) decompose'✝
      xr : Function.RightInverse (⇑(DirectSum.coeAddMonoidHom ℳ)) decompose'✝
      ⊢ Eq { decompose' := decompose'✝, left_inv := left_inv✝, right_inv := xr } y
    -/
    obtain ⟨_, yl, _⟩ := y
    /-
      case mk.mk
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      σ : Type u_4
      inst✝³ : DecidableEq ι
      inst✝² : AddCommMonoid M
      inst✝¹ : SetLike σ M
      inst✝ : AddSubmonoidClass σ M
      ℳ : ι → σ
      decompose'✝¹ : M → DirectSum ι fun i => Subtype fun x => Membership.mem (ℳ i) x
      left_inv✝ : Function.LeftInverse (⇑(DirectSum.coeAddMonoidHom ℳ)) decompose'✝¹
      xr : Function.RightInverse (⇑(DirectSum.coeAddMonoidHom ℳ)) decompose'✝¹
      decompose'✝ : M → DirectSum ι fun i => Subtype fun x => Membership.mem (ℳ i) x
      yl : Function.LeftInverse (⇑(DirectSum.coeAddMonoidHom ℳ)) decompose'✝
      right_inv✝ : Function.RightInverse (⇑(DirectSum.coeAddMonoidHom ℳ)) decompose'✝
      ⊢ Eq { decompose' := decompose'✝¹, left_inv := left_inv✝, right_inv := xr } {  …
    -/
    congr
    /-
      case mk.mk.e_decompose'
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      σ : Type u_4
      inst✝³ : DecidableEq ι
      inst✝² : AddCommMonoid M
      inst✝¹ : SetLike σ M
      inst✝ : AddSubmonoidClass σ M
      ℳ : ι → σ
      decompose'✝¹ : M → DirectSum ι fun i => Subtype fun x => Membership.mem (ℳ i) x
      left_inv✝ : Function.LeftInverse (⇑(DirectSum.coeAddMonoidHom ℳ)) decompose'✝¹
      xr : Function.RightInverse (⇑(DirectSum.coeAddMonoidHom ℳ)) decompose'✝¹
      decompose'✝ : M → DirectSum ι fun i => Subtype fun x => Membership.mem (ℳ i) x
      yl : Function.LeftInverse (⇑(DirectSum.coeAddMonoidHom ℳ)) decompose'✝
      right_inv✝ : Function.RightInverse (⇑(DirectSum.coeAddMonoidHom ℳ)) decompose'✝
      ⊢ Eq decompose'✝¹ decompose'✝
    -/
    exact Function.LeftInverse.eq_rightInverse xr yl⟩
    /-
      🎉 no goals
    -/


/-- A convenience method to construct a decomposition from an `AddMonoidHom`, such that the proofs
of left and right inverse can be constructed via `ext`. -/
abbrev Decomposition.ofAddHom (decompose : M →+ ⨁ i, ℳ i)
    (h_left_inv : (DirectSum.coeAddMonoidHom ℳ).comp decompose = .id _)
    (h_right_inv : decompose.comp (DirectSum.coeAddMonoidHom ℳ) = .id _) : Decomposition ℳ where
  decompose' := decompose
  left_inv := DFunLike.congr_fun h_left_inv
  right_inv := DFunLike.congr_fun h_right_inv


/-- Noncomputably conjure a decomposition instance from a `DirectSum.IsInternal` proof. -/
noncomputable def IsInternal.chooseDecomposition (h : IsInternal ℳ) :
    DirectSum.Decomposition ℳ where
  decompose' := (Equiv.ofBijective _ h).symm
  left_inv := (Equiv.ofBijective _ h).right_inv
  right_inv := (Equiv.ofBijective _ h).left_inv


protected theorem Decomposition.isInternal : DirectSum.IsInternal ℳ :=
  ⟨Decomposition.right_inv.injective, Decomposition.left_inv.surjective⟩


/-- If `M` is graded by `ι` with degree `i` component `ℳ i`, then it is isomorphic as
to a direct sum of components. This is the canonical spelling of the `decompose'` field. -/
def decompose : M ≃ ⨁ i, ℳ i where
  toFun := Decomposition.decompose'
  invFun := DirectSum.coeAddMonoidHom ℳ
  left_inv := Decomposition.left_inv
  right_inv := Decomposition.right_inv


protected theorem Decomposition.inductionOn {p : M → Prop} (h_zero : p 0)
    (h_homogeneous : ∀ {i} (m : ℳ i), p (m : M)) (h_add : ∀ m m' : M, p m → p m' → p (m + m')) :
    ∀ m, p m := by
  let ℳ' : ι → AddSubmonoid M := fun i ↦
    (⟨⟨ℳ i, fun x y ↦ AddMemClass.add_mem x y⟩, (ZeroMemClass.zero_mem _)⟩ : AddSubmonoid M)
  haveI t : DirectSum.Decomposition ℳ' :=
    { decompose' := DirectSum.decompose ℳ
      left_inv := fun _ ↦ (decompose ℳ).left_inv _
      right_inv := fun _ ↦ (decompose ℳ).right_inv _ }
  have mem : ∀ m, m ∈ iSup ℳ' := fun _m ↦
    (DirectSum.IsInternal.addSubmonoid_iSup_eq_top ℳ' (Decomposition.isInternal ℳ')).symm ▸ trivial
  -- Porting note: needs to use @ even though no implicit argument is provided
  exact fun m ↦ @AddSubmonoid.iSup_induction _ _ _ ℳ' _ _ (mem m)
    (fun i m h ↦ h_homogeneous ⟨m, h⟩) h_zero h_add
--  exact fun m ↦
--    AddSubmonoid.iSup_induction ℳ' (mem m) (fun i m h ↦ h_homogeneous ⟨m, h⟩) h_zero h_add


@[simp]
theorem Decomposition.decompose'_eq : Decomposition.decompose' = decompose ℳ := rfl


@[simp]
theorem decompose_symm_of {i : ι} (x : ℳ i) : (decompose ℳ).symm (DirectSum.of _ i x) = x :=
  DirectSum.coeAddMonoidHom_of ℳ _ _


@[simp]
theorem decompose_coe {i : ι} (x : ℳ i) : decompose ℳ (x : M) = DirectSum.of _ i x := by
  /-
    ι : Type u_1
    M : Type u_3
    σ : Type u_4
    inst✝⁴ : DecidableEq ι
    inst✝³ : AddCommMonoid M
    inst✝² : SetLike σ M
    inst✝¹ : AddSubmonoidClass σ M
    ℳ : ι → σ
    inst✝ : DirectSum.Decomposition ℳ
    i : ι
    x : Subtype fun x => Membership.mem (ℳ i) x
    ⊢ Eq ((DirectSum.decompose ℳ) ↑x) ((DirectSum.of (fun i => Subtype fun x => Me …
  -/
  rw [← decompose_symm_of _, Equiv.apply_symm_apply]
  /-
    🎉 no goals
  -/


theorem decompose_of_mem {x : M} {i : ι} (hx : x ∈ ℳ i) :
    decompose ℳ x = DirectSum.of (fun i ↦ ℳ i) i ⟨x, hx⟩ :=
  decompose_coe _ ⟨x, hx⟩


theorem decompose_of_mem_same {x : M} {i : ι} (hx : x ∈ ℳ i) : (decompose ℳ x i : M) = x := by
  /-
    ι : Type u_1
    M : Type u_3
    σ : Type u_4
    inst✝⁴ : DecidableEq ι
    inst✝³ : AddCommMonoid M
    inst✝² : SetLike σ M
    inst✝¹ : AddSubmonoidClass σ M
    ℳ : ι → σ
    inst✝ : DirectSum.Decomposition ℳ
    x : M
    i : ι
    hx : Membership.mem (ℳ i) x
    ⊢ Eq (↑(((DirectSum.decompose ℳ) x) i)) x
  -/
  rw [decompose_of_mem _ hx, DirectSum.of_eq_same, Subtype.coe_mk]
  /-
    🎉 no goals
  -/


theorem decompose_of_mem_ne {x : M} {i j : ι} (hx : x ∈ ℳ i) (hij : i ≠ j) :
    (decompose ℳ x j : M) = 0 := by
  /-
    ι : Type u_1
    M : Type u_3
    σ : Type u_4
    inst✝⁴ : DecidableEq ι
    inst✝³ : AddCommMonoid M
    inst✝² : SetLike σ M
    inst✝¹ : AddSubmonoidClass σ M
    ℳ : ι → σ
    inst✝ : DirectSum.Decomposition ℳ
    x : M
    i j : ι
    hx : Membership.mem (ℳ i) x
    hij : Ne i j
    ⊢ Eq (↑(((DirectSum.decompose ℳ) x) j)) 0
  -/
  rw [decompose_of_mem _ hx, DirectSum.of_eq_of_ne _ _ _ hij, ZeroMemClass.coe_zero]
  /-
    🎉 no goals
  -/


theorem degree_eq_of_mem_mem {x : M} {i j : ι} (hxi : x ∈ ℳ i) (hxj : x ∈ ℳ j) (hx : x ≠ 0) :
    i = j := by
  /-
    ι : Type u_1
    M : Type u_3
    σ : Type u_4
    inst✝⁴ : DecidableEq ι
    inst✝³ : AddCommMonoid M
    inst✝² : SetLike σ M
    inst✝¹ : AddSubmonoidClass σ M
    ℳ : ι → σ
    inst✝ : DirectSum.Decomposition ℳ
    x : M
    i j : ι
    hxi : Membership.mem (ℳ i) x
    hxj : Membership.mem (ℳ j) x
    hx : Ne x 0
    ⊢ Eq i j
  -/
  contrapose! hx; rw [← decompose_of_mem_same ℳ hxj, decompose_of_mem_ne ℳ hxi hx]
                  /-
                    🎉 no goals
                  -/


/-- If `M` is graded by `ι` with degree `i` component `ℳ i`, then it is isomorphic as
an additive monoid to a direct sum of components. -/
-- Porting note: deleted [simps] and added the corresponding lemmas by hand
def decomposeAddEquiv : M ≃+ ⨁ i, ℳ i :=
  AddEquiv.symm { (decompose ℳ).symm with map_add' := map_add (DirectSum.coeAddMonoidHom ℳ) }


@[simp]
lemma decomposeAddEquiv_apply (a : M) :
    decomposeAddEquiv ℳ a = decompose ℳ a := rfl


@[simp]
lemma decomposeAddEquiv_symm_apply (a : ⨁ i, ℳ i) :
    (decomposeAddEquiv ℳ).symm a = (decompose ℳ).symm a := rfl


@[simp]
theorem decompose_zero : decompose ℳ (0 : M) = 0 :=
  map_zero (decomposeAddEquiv ℳ)


@[simp]
theorem decompose_symm_zero : (decompose ℳ).symm 0 = (0 : M) :=
  map_zero (decomposeAddEquiv ℳ).symm


@[simp]
theorem decompose_add (x y : M) : decompose ℳ (x + y) = decompose ℳ x + decompose ℳ y :=
  map_add (decomposeAddEquiv ℳ) x y


@[simp]
theorem decompose_symm_add (x y : ⨁ i, ℳ i) :
    (decompose ℳ).symm (x + y) = (decompose ℳ).symm x + (decompose ℳ).symm y :=
  map_add (decomposeAddEquiv ℳ).symm x y


@[simp]
theorem decompose_sum {ι'} (s : Finset ι') (f : ι' → M) :
    decompose ℳ (∑ i ∈ s, f i) = ∑ i ∈ s, decompose ℳ (f i) :=
  map_sum (decomposeAddEquiv ℳ) f s


@[simp]
theorem decompose_symm_sum {ι'} (s : Finset ι') (f : ι' → ⨁ i, ℳ i) :
    (decompose ℳ).symm (∑ i ∈ s, f i) = ∑ i ∈ s, (decompose ℳ).symm (f i) :=
  map_sum (decomposeAddEquiv ℳ).symm f s


theorem sum_support_decompose [∀ (i) (x : ℳ i), Decidable (x ≠ 0)] (r : M) :
    (∑ i ∈ (decompose ℳ r).support, (decompose ℳ r i : M)) = r := by
  conv_rhs =>
    rw [← (decompose ℳ).symm_apply_apply r, ← sum_support_of (decompose ℳ r)]
  /-
    ι : Type u_1
    M : Type u_3
    σ : Type u_4
    inst✝⁵ : DecidableEq ι
    inst✝⁴ : AddCommMonoid M
    inst✝³ : SetLike σ M
    inst✝² : AddSubmonoidClass σ M
    ℳ : ι → σ
    inst✝¹ : DirectSum.Decomposition ℳ
    inst✝ : (i : ι) → (x : Subtype fun x => Membership.mem (ℳ i) x) → Decidable (N …
    r : M
    ⊢ Eq ((DFinsupp.support ((DirectSum.decompose ℳ) r)).sum fun i => ↑(((DirectSu …
  -/
  rw [decompose_symm_sum]
  /-
    ι : Type u_1
    M : Type u_3
    σ : Type u_4
    inst✝⁵ : DecidableEq ι
    inst✝⁴ : AddCommMonoid M
    inst✝³ : SetLike σ M
    inst✝² : AddSubmonoidClass σ M
    ℳ : ι → σ
    inst✝¹ : DirectSum.Decomposition ℳ
    inst✝ : (i : ι) → (x : Subtype fun x => Membership.mem (ℳ i) x) → Decidable (N …
    r : M
    ⊢ Eq ((DFinsupp.support ((DirectSum.decompose ℳ) r)).sum fun i => ↑(((DirectSu …
  -/
  simp_rw [decompose_symm_of]
  /-
    🎉 no goals
  -/


/-- The `-` in the statements below doesn't resolve without this line.

This seems to be a problem of synthesized vs inferred typeclasses disagreeing. If we replace
the statement of `decompose_neg` with `@Eq (⨁ i, ℳ i) (decompose ℳ (-x)) (-decompose ℳ x)`
instead of `decompose ℳ (-x) = -decompose ℳ x`, which forces the typeclasses needed by `⨁ i, ℳ i`
to be found by unification rather than synthesis, then everything works fine without this
instance. -/
instance addCommGroupSetLike [AddCommGroup M] [SetLike σ M] [AddSubgroupClass σ M] (ℳ : ι → σ) :
                                  /-
                                    ι : Type u_1
                                    R : Type u_2
                                    M : Type u_3
                                    σ : Type u_4
                                    inst✝² : AddCommGroup M
                                    inst✝¹ : SetLike σ M
                                    inst✝ : AddSubgroupClass σ M
                                    ℳ : ι → σ
                                    ⊢ AddCommGroup (DirectSum ι fun i => Subtype fun x => Membership.mem (ℳ i) x)
                                  -/
    AddCommGroup (⨁ i, ℳ i) := by infer_instance
                                  /-
                                    🎉 no goals
                                  -/


@[simp]
theorem decompose_neg (x : M) : decompose ℳ (-x) = -decompose ℳ x :=
  map_neg (decomposeAddEquiv ℳ) x


@[simp]
theorem decompose_symm_neg (x : ⨁ i, ℳ i) : (decompose ℳ).symm (-x) = -(decompose ℳ).symm x :=
  map_neg (decomposeAddEquiv ℳ).symm x


@[simp]
theorem decompose_sub (x y : M) : decompose ℳ (x - y) = decompose ℳ x - decompose ℳ y :=
  map_sub (decomposeAddEquiv ℳ) x y


@[simp]
theorem decompose_symm_sub (x y : ⨁ i, ℳ i) :
    (decompose ℳ).symm (x - y) = (decompose ℳ).symm x - (decompose ℳ).symm y :=
  map_sub (decomposeAddEquiv ℳ).symm x y


/-- A convenience method to construct a decomposition from an `LinearMap`, such that the proofs
of left and right inverse can be constructed via `ext`. -/
abbrev Decomposition.ofLinearMap (decompose : M →ₗ[R] ⨁ i, ℳ i)
    (h_left_inv : DirectSum.coeLinearMap ℳ ∘ₗ decompose = .id)
    (h_right_inv : decompose ∘ₗ DirectSum.coeLinearMap ℳ = .id) : Decomposition ℳ where
  decompose' := decompose
  left_inv := DFunLike.congr_fun h_left_inv
  right_inv := DFunLike.congr_fun h_right_inv


/-- If `M` is graded by `ι` with degree `i` component `ℳ i`, then it is isomorphic as
a module to a direct sum of components. -/
def decomposeLinearEquiv : M ≃ₗ[R] ⨁ i, ℳ i :=
  LinearEquiv.symm
    { (decomposeAddEquiv ℳ).symm with map_smul' := map_smul (DirectSum.coeLinearMap ℳ) }


@[simp] theorem decomposeLinearEquiv_apply (m : M) :
    decomposeLinearEquiv ℳ m = decompose ℳ m := rfl


@[simp] theorem decomposeLinearEquiv_symm_apply (m : ⨁ i, ℳ i) :
    (decomposeLinearEquiv ℳ).symm m = (decompose ℳ).symm m := rfl


@[simp]
theorem decompose_smul (r : R) (x : M) : decompose ℳ (r • x) = r • decompose ℳ x :=
  map_smul (decomposeLinearEquiv ℳ) r x


@[simp] theorem decomposeLinearEquiv_symm_comp_lof (i : ι) :
    (decomposeLinearEquiv ℳ).symm ∘ₗ lof R ι (ℳ ·) i = (ℳ i).subtype :=
  LinearMap.ext <| decompose_symm_of _


/-- Two linear maps from a module with a decomposition agree if they agree on every piece.

Note this cannot be `@[ext]` as `ℳ` cannot be inferred. -/
theorem decompose_lhom_ext {N} [AddCommMonoid N] [Module R N] ⦃f g : M →ₗ[R] N⦄
    (h : ∀ i, f ∘ₗ (ℳ i).subtype = g ∘ₗ (ℳ i).subtype) : f = g :=
  LinearMap.ext <| (decomposeLinearEquiv ℳ).symm.surjective.forall.mpr <|
    suffices f ∘ₗ (decomposeLinearEquiv ℳ).symm
           = (g ∘ₗ (decomposeLinearEquiv ℳ).symm : (⨁ i, ℳ i) →ₗ[R] N) from
      DFunLike.congr_fun this
    linearMap_ext _ fun i => by
      /-
        ι : Type u_1
        R : Type u_2
        M : Type u_3
        inst✝⁶ : DecidableEq ι
        inst✝⁵ : Semiring R
        inst✝⁴ : AddCommMonoid M
        inst✝³ : Module R M
        ℳ : ι → Submodule R M
        inst✝² : DirectSum.Decomposition ℳ
        N : Type u_5
        inst✝¹ : AddCommMonoid N
        inst✝ : Module R N
        f g : LinearMap (RingHom.id R) M N
        h : ∀ (i : ι), Eq (f.comp (ℳ i).subtype) (g.comp (ℳ i).subtype)
        i : ι
        ⊢ Eq ((f.comp ↑(DirectSum.decomposeLinearEquiv ℳ).symm).comp (DirectSum.lof R  …
      -/
      simp_rw [LinearMap.comp_assoc, decomposeLinearEquiv_symm_comp_lof ℳ i, h]
      /-
        🎉 no goals
      -/


