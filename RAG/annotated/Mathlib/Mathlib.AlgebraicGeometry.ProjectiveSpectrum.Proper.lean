lemma lift_awayMapₐ_awayMapₐ_surjective {d e : ℕ} {f : A} (hf : f ∈ 𝒜 d)
    {g : A} (hg : g ∈ 𝒜 e) {x : A} (hx : x = f * g) (hd : 0 < d) :
    Function.Surjective
      (Algebra.TensorProduct.lift (awayMapₐ 𝒜 hg hx) (awayMapₐ 𝒜 hf (hx.trans (mul_comm _ _)))
        (fun _ _ ↦ .all _ _)) := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    d e : Nat
    f : A
    hf : Membership.mem (𝒜 d) f
    g : A
    hg : Membership.mem (𝒜 e) g
    x : A
    hx : Eq x (HMul.hMul f g)
    hd : LT.lt 0 d
    ⊢ Function.Surjective ⇑(Algebra.TensorProduct.lift (HomogeneousLocalization.aw …
  -/
  intro z
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    d e : Nat
    f : A
    hf : Membership.mem (𝒜 d) f
    g : A
    hg : Membership.mem (𝒜 e) g
    x : A
    hx : Eq x (HMul.hMul f g)
    hd : LT.lt 0 d
    z : HomogeneousLocalization.Away 𝒜 x
    ⊢ Exists fun a => Eq ((Algebra.TensorProduct.lift (HomogeneousLocalization.awa …
  -/
  obtain ⟨⟨n, ⟨a, ha⟩, ⟨b, hb'⟩, ⟨j, (rfl : _ = b)⟩⟩, rfl⟩ := mk_surjective z
  /-
    case intro.mk.mk.mk.intro
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    d e : Nat
    f : A
    hf : Membership.mem (𝒜 d) f
    g : A
    hg : Membership.mem (𝒜 e) g
    x : A
    hx : Eq x (HMul.hMul f g)
    hd : LT.lt 0 d
    n : Nat
    a : A
    ha : Membership.mem (𝒜 n) a
    j : Nat
    hb' : Membership.mem (𝒜 n) ((fun x_1 => HPow.hPow x x_1) j)
    ⊢ Exists fun a_1 => Eq ((Algebra.TensorProduct.lift (HomogeneousLocalization.a …
  -/
  by_cases hfg : (f * g) ^ j = 0
    /-
      case pos
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      d e : Nat
      f : A
      hf : Membership.mem (𝒜 d) f
      g : A
      hg : Membership.mem (𝒜 e) g
      x : A
      hx : Eq x (HMul.hMul f g)
      hd : LT.lt 0 d
      n : Nat
      a : A
      ha : Membership.mem (𝒜 n) a
      j : Nat
      hb' : Membership.mem (𝒜 n) ((fun x_1 => HPow.hPow x x_1) j)
      hfg : Eq (HPow.hPow (HMul.hMul f g) j) 0
      ⊢ Exists fun a_1 => Eq ((Algebra.TensorProduct.lift (HomogeneousLocalization.a …
    -/
  · use 0
    /-
      case h
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      d e : Nat
      f : A
      hf : Membership.mem (𝒜 d) f
      g : A
      hg : Membership.mem (𝒜 e) g
      x : A
      hx : Eq x (HMul.hMul f g)
      hd : LT.lt 0 d
      n : Nat
      a : A
      ha : Membership.mem (𝒜 n) a
      j : Nat
      hb' : Membership.mem (𝒜 n) ((fun x_1 => HPow.hPow x x_1) j)
      hfg : Eq (HPow.hPow (HMul.hMul f g) j) 0
      ⊢ Eq ((Algebra.TensorProduct.lift (HomogeneousLocalization.awayMapₐ 𝒜 hg hx) ( …
    -/
    have := HomogeneousLocalization.subsingleton 𝒜 (x := Submonoid.powers x) ⟨j, hx ▸ hfg⟩
    /-
      case h
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      d e : Nat
      f : A
      hf : Membership.mem (𝒜 d) f
      g : A
      hg : Membership.mem (𝒜 e) g
      x : A
      hx : Eq x (HMul.hMul f g)
      hd : LT.lt 0 d
      n : Nat
      a : A
      ha : Membership.mem (𝒜 n) a
      j : Nat
      hb' : Membership.mem (𝒜 n) ((fun x_1 => HPow.hPow x x_1) j)
      hfg : Eq (HPow.hPow (HMul.hMul f g) j) 0
      this : Subsingleton (HomogeneousLocalization 𝒜 (Submonoid.powers x))
      ⊢ Eq ((Algebra.TensorProduct.lift (HomogeneousLocalization.awayMapₐ 𝒜 hg hx) ( …
    -/
    exact this.elim _ _
    /-
      🎉 no goals
    -/
  have : n = j * (d + e) := by
    apply DirectSum.degree_eq_of_mem_mem 𝒜 hb'
    convert SetLike.pow_mem_graded _ _ using 2
    · infer_instance
    · exact hx ▸ SetLike.mul_mem_graded hf hg
    · exact hx ▸ hfg
  let x0 : NumDenSameDeg 𝒜 (.powers f) :=
  { deg := j * (d * (e + 1))
    num := ⟨a * g ^ (j * (d - 1)), by
      convert SetLike.mul_mem_graded ha (SetLike.pow_mem_graded _ hg) using 2
      rw [this]
      cases d
      · contradiction
      · simp; ring⟩
    den := ⟨f ^ (j * (e + 1)), by convert SetLike.pow_mem_graded _ hf using 2; ring⟩
    den_mem := ⟨_,rfl⟩ }
  let y0 : NumDenSameDeg 𝒜 (.powers g) :=
  { deg := j * (d * e)
    num := ⟨f ^ (j * e), by convert SetLike.pow_mem_graded _ hf using 2; ring⟩
    den := ⟨g ^ (j * d), by convert SetLike.pow_mem_graded _ hg using 2; ring⟩
    den_mem := ⟨_,rfl⟩ }
  /-
    case neg
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    d e : Nat
    f : A
    hf : Membership.mem (𝒜 d) f
    g : A
    hg : Membership.mem (𝒜 e) g
    x : A
    hx : Eq x (HMul.hMul f g)
    hd : LT.lt 0 d
    n : Nat
    a : A
    ha : Membership.mem (𝒜 n) a
    j : Nat
    hb' : Membership.mem (𝒜 n) ((fun x_1 => HPow.hPow x x_1) j)
    hfg : Not (Eq (HPow.hPow (HMul.hMul f g) j) 0)
    this : Eq n (HMul.hMul j (HAdd.hAdd d e))
    x0 : HomogeneousLocalization.NumDenSameDeg 𝒜 (Submonoid.powers f) := { deg :=  …
    y0 : HomogeneousLocalization.NumDenSameDeg 𝒜 (Submonoid.powers g) := { deg :=  …
    ⊢ Exists fun a_1 => Eq ((Algebra.TensorProduct.lift (HomogeneousLocalization.a …
  -/
  use mk x0 ⊗ₜ mk y0
  /-
    case h
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    d e : Nat
    f : A
    hf : Membership.mem (𝒜 d) f
    g : A
    hg : Membership.mem (𝒜 e) g
    x : A
    hx : Eq x (HMul.hMul f g)
    hd : LT.lt 0 d
    n : Nat
    a : A
    ha : Membership.mem (𝒜 n) a
    j : Nat
    hb' : Membership.mem (𝒜 n) ((fun x_1 => HPow.hPow x x_1) j)
    hfg : Not (Eq (HPow.hPow (HMul.hMul f g) j) 0)
    this : Eq n (HMul.hMul j (HAdd.hAdd d e))
    x0 : HomogeneousLocalization.NumDenSameDeg 𝒜 (Submonoid.powers f) := { deg :=  …
    y0 : HomogeneousLocalization.NumDenSameDeg 𝒜 (Submonoid.powers g) := { deg :=  …
    ⊢ Eq ((Algebra.TensorProduct.lift (HomogeneousLocalization.awayMapₐ 𝒜 hg hx) ( …
  -/
  ext
  simp only [Algebra.TensorProduct.lift_tmul, awayMapₐ_apply, val_mul,
    val_awayMap_mk, Localization.mk_mul, val_mk, x0, y0]
  /-
    case h.a
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    d e : Nat
    f : A
    hf : Membership.mem (𝒜 d) f
    g : A
    hg : Membership.mem (𝒜 e) g
    x : A
    hx : Eq x (HMul.hMul f g)
    hd : LT.lt 0 d
    n : Nat
    a : A
    ha : Membership.mem (𝒜 n) a
    j : Nat
    hb' : Membership.mem (𝒜 n) ((fun x_1 => HPow.hPow x x_1) j)
    hfg : Not (Eq (HPow.hPow (HMul.hMul f g) j) 0)
    this : Eq n (HMul.hMul j (HAdd.hAdd d e))
    x0 : HomogeneousLocalization.NumDenSameDeg 𝒜 (Submonoid.powers f) := { deg :=  …
    y0 : HomogeneousLocalization.NumDenSameDeg 𝒜 (Submonoid.powers g) := { deg :=  …
    ⊢ Eq (Localization.mk (HMul.hMul (HMul.hMul (HMul.hMul a (HPow.hPow g (HMul.hM …
  -/
  rw [Localization.mk_eq_mk_iff, Localization.r_iff_exists]
  /-
    case h.a
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    d e : Nat
    f : A
    hf : Membership.mem (𝒜 d) f
    g : A
    hg : Membership.mem (𝒜 e) g
    x : A
    hx : Eq x (HMul.hMul f g)
    hd : LT.lt 0 d
    n : Nat
    a : A
    ha : Membership.mem (𝒜 n) a
    j : Nat
    hb' : Membership.mem (𝒜 n) ((fun x_1 => HPow.hPow x x_1) j)
    hfg : Not (Eq (HPow.hPow (HMul.hMul f g) j) 0)
    this : Eq n (HMul.hMul j (HAdd.hAdd d e))
    x0 : HomogeneousLocalization.NumDenSameDeg 𝒜 (Submonoid.powers f) := { deg :=  …
    y0 : HomogeneousLocalization.NumDenSameDeg 𝒜 (Submonoid.powers g) := { deg :=  …
    ⊢ Exists fun c => Eq (HMul.hMul (↑c) (HMul.hMul ↑{ fst := a, snd := ⟨HPow.hPow …
  -/
  use 1
  /-
    case h
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    d e : Nat
    f : A
    hf : Membership.mem (𝒜 d) f
    g : A
    hg : Membership.mem (𝒜 e) g
    x : A
    hx : Eq x (HMul.hMul f g)
    hd : LT.lt 0 d
    n : Nat
    a : A
    ha : Membership.mem (𝒜 n) a
    j : Nat
    hb' : Membership.mem (𝒜 n) ((fun x_1 => HPow.hPow x x_1) j)
    hfg : Not (Eq (HPow.hPow (HMul.hMul f g) j) 0)
    this : Eq n (HMul.hMul j (HAdd.hAdd d e))
    x0 : HomogeneousLocalization.NumDenSameDeg 𝒜 (Submonoid.powers f) := { deg :=  …
    y0 : HomogeneousLocalization.NumDenSameDeg 𝒜 (Submonoid.powers g) := { deg :=  …
    ⊢ Eq (HMul.hMul (↑1) (HMul.hMul ↑{ fst := a, snd := ⟨HPow.hPow x j, ⋯⟩ }.2 { f …
  -/
  simp only [OneMemClass.coe_one, one_mul, Submonoid.mk_mul_mk]
  /-
    case h
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    d e : Nat
    f : A
    hf : Membership.mem (𝒜 d) f
    g : A
    hg : Membership.mem (𝒜 e) g
    x : A
    hx : Eq x (HMul.hMul f g)
    hd : LT.lt 0 d
    n : Nat
    a : A
    ha : Membership.mem (𝒜 n) a
    j : Nat
    hb' : Membership.mem (𝒜 n) ((fun x_1 => HPow.hPow x x_1) j)
    hfg : Not (Eq (HPow.hPow (HMul.hMul f g) j) 0)
    this : Eq n (HMul.hMul j (HAdd.hAdd d e))
    x0 : HomogeneousLocalization.NumDenSameDeg 𝒜 (Submonoid.powers f) := { deg :=  …
    y0 : HomogeneousLocalization.NumDenSameDeg 𝒜 (Submonoid.powers g) := { deg :=  …
    ⊢ Eq (HMul.hMul (HPow.hPow x j) (HMul.hMul (HMul.hMul (HMul.hMul a (HPow.hPow  …
  -/
  cases d
    /-
      case h.zero
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      e : Nat
      f g : A
      hg : Membership.mem (𝒜 e) g
      x : A
      hx : Eq x (HMul.hMul f g)
      n : Nat
      a : A
      ha : Membership.mem (𝒜 n) a
      j : Nat
      hb' : Membership.mem (𝒜 n) ((fun x_1 => HPow.hPow x x_1) j)
      hfg : Not (Eq (HPow.hPow (HMul.hMul f g) j) 0)
      hf : Membership.mem (𝒜 0) f
      hd : LT.lt 0 0
      this : Eq n (HMul.hMul j (HAdd.hAdd 0 e))
      x0 : HomogeneousLocalization.NumDenSameDeg 𝒜 (Submonoid.powers f) := { deg :=  …
      y0 : HomogeneousLocalization.NumDenSameDeg 𝒜 (Submonoid.powers g) := { deg :=  …
      ⊢ Eq (HMul.hMul (HPow.hPow x j) (HMul.hMul (HMul.hMul (HMul.hMul a (HPow.hPow  …
    -/
  · contradiction
    /-
      🎉 no goals
    -/
    /-
      case h.succ
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      e : Nat
      f g : A
      hg : Membership.mem (𝒜 e) g
      x : A
      hx : Eq x (HMul.hMul f g)
      n : Nat
      a : A
      ha : Membership.mem (𝒜 n) a
      j : Nat
      hb' : Membership.mem (𝒜 n) ((fun x_1 => HPow.hPow x x_1) j)
      hfg : Not (Eq (HPow.hPow (HMul.hMul f g) j) 0)
      n✝ : Nat
      hf : Membership.mem (𝒜 (HAdd.hAdd n✝ 1)) f
      hd : LT.lt 0 (HAdd.hAdd n✝ 1)
      this : Eq n (HMul.hMul j (HAdd.hAdd (HAdd.hAdd n✝ 1) e))
      x0 : HomogeneousLocalization.NumDenSameDeg 𝒜 (Submonoid.powers f) := { deg :=  …
      y0 : HomogeneousLocalization.NumDenSameDeg 𝒜 (Submonoid.powers g) := { deg :=  …
      ⊢ Eq (HMul.hMul (HPow.hPow x j) (HMul.hMul (HMul.hMul (HMul.hMul a (HPow.hPow  …
    -/
  · simp only [hx, add_tsub_cancel_right]
    /-
      case h.succ
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      e : Nat
      f g : A
      hg : Membership.mem (𝒜 e) g
      x : A
      hx : Eq x (HMul.hMul f g)
      n : Nat
      a : A
      ha : Membership.mem (𝒜 n) a
      j : Nat
      hb' : Membership.mem (𝒜 n) ((fun x_1 => HPow.hPow x x_1) j)
      hfg : Not (Eq (HPow.hPow (HMul.hMul f g) j) 0)
      n✝ : Nat
      hf : Membership.mem (𝒜 (HAdd.hAdd n✝ 1)) f
      hd : LT.lt 0 (HAdd.hAdd n✝ 1)
      this : Eq n (HMul.hMul j (HAdd.hAdd (HAdd.hAdd n✝ 1) e))
      x0 : HomogeneousLocalization.NumDenSameDeg 𝒜 (Submonoid.powers f) := { deg :=  …
      y0 : HomogeneousLocalization.NumDenSameDeg 𝒜 (Submonoid.powers g) := { deg :=  …
      ⊢ Eq (HMul.hMul (HPow.hPow (HMul.hMul f g) j) (HMul.hMul (HMul.hMul (HMul.hMul …
    -/
    ring
    /-
      🎉 no goals
    -/


open TensorProduct in
instance isSeparated : IsSeparated (toSpecZero 𝒜) := by
  refine ⟨IsLocalAtTarget.of_openCover (Pullback.openCoverOfLeftRight
    (affineOpenCover 𝒜).openCover (affineOpenCover 𝒜).openCover _ _) ?_⟩
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    ⊢ ∀ (i : (AlgebraicGeometry.Scheme.Pullback.openCoverOfLeftRight (AlgebraicGeo …
  -/
  intro ⟨i, j⟩
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    i j : (AlgebraicGeometry.Proj.affineOpenCover 𝒜).openCover.J
    ⊢ AlgebraicGeometry.IsClosedImmersion (AlgebraicGeometry.Scheme.Cover.pullback …
  -/
  dsimp [Scheme, Cover.pullbackHom]
  refine (MorphismProperty.cancel_left_of_respectsIso (P := @IsClosedImmersion)
    (f := (pullbackDiagonalMapIdIso ..).inv) _).mp ?_
  let e₁ : pullback ((affineOpenCover 𝒜).map i ≫ toSpecZero 𝒜)
        ((affineOpenCover 𝒜).map j ≫ toSpecZero 𝒜) ≅
        Spec (.of <| TensorProduct (𝒜 0) (Away 𝒜 i.2) (Away 𝒜 j.2)) := by
    refine pullback.congrHom ?_ ?_ ≪≫ pullbackSpecIso (𝒜 0) (Away 𝒜 i.2) (Away 𝒜 j.2)
    · simp [affineOpenCover, openCoverOfISupEqTop, awayι_toSpecZero]; rfl
    · simp [affineOpenCover, openCoverOfISupEqTop, awayι_toSpecZero]; rfl
  let e₂ : pullback ((affineOpenCover 𝒜).map i) ((affineOpenCover 𝒜).map j) ≅
        Spec (.of <| (Away 𝒜 (i.2 * j.2))) :=
    pullbackAwayιIso 𝒜 _ _ _ _ rfl
  rw [← MorphismProperty.cancel_right_of_respectsIso (P := @IsClosedImmersion) _ e₁.hom,
    ← MorphismProperty.cancel_left_of_respectsIso (P := @IsClosedImmersion) e₂.inv]
  let F : Away 𝒜 i.2.1 ⊗[𝒜 0] Away 𝒜 j.2.1 →+* Away 𝒜 (i.2.1 * j.2.1) :=
    (Algebra.TensorProduct.lift (awayMapₐ 𝒜 j.2.2 rfl) (awayMapₐ 𝒜 i.2.2 (mul_comm _ _))
      (fun _ _ ↦ .all _ _)).toRingHom
  /-
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    i j : (AlgebraicGeometry.Proj.affineOpenCover 𝒜).openCover.J
    e₁ : CategoryTheory.Iso (CategoryTheory.Limits.pullback (CategoryTheory.Catego …
    e₂ : CategoryTheory.Iso (CategoryTheory.Limits.pullback ((AlgebraicGeometry.Pr …
    F : RingHom (TensorProduct (Subtype fun x => Membership.mem (𝒜 0) x) (Homogene …
    ⊢ AlgebraicGeometry.IsClosedImmersion (CategoryTheory.CategoryStruct.comp e₂.i …
  -/
  have : Function.Surjective F := lift_awayMapₐ_awayMapₐ_surjective 𝒜 i.2.2 j.2.2 rfl i.1.2
  convert IsClosedImmersion.spec_of_surjective
    (CommRingCat.ofHom (R := Away 𝒜 i.2.1 ⊗[𝒜 0] Away 𝒜 j.2.1) F) this using 1
  /-
    case h.e'_3
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    i j : (AlgebraicGeometry.Proj.affineOpenCover 𝒜).openCover.J
    e₁ : CategoryTheory.Iso (CategoryTheory.Limits.pullback (CategoryTheory.Catego …
    e₂ : CategoryTheory.Iso (CategoryTheory.Limits.pullback ((AlgebraicGeometry.Pr …
    F : RingHom (TensorProduct (Subtype fun x => Membership.mem (𝒜 0) x) (Homogene …
    this : Function.Surjective ⇑F
    ⊢ Eq (CategoryTheory.CategoryStruct.comp e₂.inv (CategoryTheory.CategoryStruct …
  -/
  rw [← cancel_mono (pullbackSpecIso ..).inv]
  /-
    case h.e'_3
    R : Type u_1
    A : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    𝒜 : Nat → Submodule R A
    inst✝ : GradedAlgebra 𝒜
    i j : (AlgebraicGeometry.Proj.affineOpenCover 𝒜).openCover.J
    e₁ : CategoryTheory.Iso (CategoryTheory.Limits.pullback (CategoryTheory.Catego …
    e₂ : CategoryTheory.Iso (CategoryTheory.Limits.pullback ((AlgebraicGeometry.Pr …
    F : RingHom (TensorProduct (Subtype fun x => Membership.mem (𝒜 0) x) (Homogene …
    this : Function.Surjective ⇑F
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp e …
  -/
  apply pullback.hom_ext
  · simp only [Iso.trans_hom, congrHom_hom, Category.assoc, Iso.hom_inv_id, Category.comp_id,
      limit.lift_π, id_eq, eq_mpr_eq_cast, PullbackCone.mk_pt, PullbackCone.mk_π_app, e₂, e₁,
      pullbackDiagonalMapIdIso_inv_snd_fst, AlgHom.toRingHom_eq_coe, pullbackSpecIso_inv_fst,
      ← Spec.map_comp]
    /-
      case h.e'_3.h₀
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      i j : (AlgebraicGeometry.Proj.affineOpenCover 𝒜).openCover.J
      e₁ : CategoryTheory.Iso (CategoryTheory.Limits.pullback (CategoryTheory.Catego …
      e₂ : CategoryTheory.Iso (CategoryTheory.Limits.pullback ((AlgebraicGeometry.Pr …
      F : RingHom (TensorProduct (Subtype fun x => Membership.mem (𝒜 0) x) (Homogene …
      this : Function.Surjective ⇑F
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Proj.pullbackAwayι …
    -/
    erw [pullbackAwayιIso_inv_fst]
    /-
      case h.e'_3.h₀
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      i j : (AlgebraicGeometry.Proj.affineOpenCover 𝒜).openCover.J
      e₁ : CategoryTheory.Iso (CategoryTheory.Limits.pullback (CategoryTheory.Catego …
      e₂ : CategoryTheory.Iso (CategoryTheory.Limits.pullback ((AlgebraicGeometry.Pr …
      F : RingHom (TensorProduct (Subtype fun x => Membership.mem (𝒜 0) x) (Homogene …
      this : Function.Surjective ⇑F
      ⊢ Eq (AlgebraicGeometry.Spec.map (CommRingCat.ofHom (HomogeneousLocalization.a …
    -/
    congr 1
    /-
      case h.e'_3.h₀.e_f
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      i j : (AlgebraicGeometry.Proj.affineOpenCover 𝒜).openCover.J
      e₁ : CategoryTheory.Iso (CategoryTheory.Limits.pullback (CategoryTheory.Catego …
      e₂ : CategoryTheory.Iso (CategoryTheory.Limits.pullback ((AlgebraicGeometry.Pr …
      F : RingHom (TensorProduct (Subtype fun x => Membership.mem (𝒜 0) x) (Homogene …
      this : Function.Surjective ⇑F
      ⊢ Eq (CommRingCat.ofHom (HomogeneousLocalization.awayMap 𝒜 ⋯ ⋯)) (CategoryTheo …
    -/
    ext x : 2
    exact DFunLike.congr_fun (Algebra.TensorProduct.lift_comp_includeLeft
      (awayMapₐ 𝒜 j.2.2 rfl) (awayMapₐ 𝒜 i.2.2 (mul_comm _ _)) (fun _ _ ↦ .all _ _)).symm x
  · simp only [Iso.trans_hom, congrHom_hom, Category.assoc, Iso.hom_inv_id, Category.comp_id,
      limit.lift_π, id_eq, eq_mpr_eq_cast, PullbackCone.mk_pt, PullbackCone.mk_π_app,
      pullbackDiagonalMapIdIso_inv_snd_snd, AlgHom.toRingHom_eq_coe, pullbackSpecIso_inv_snd, ←
      Spec.map_comp, e₂, e₁]
    /-
      case h.e'_3.h₁
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      i j : (AlgebraicGeometry.Proj.affineOpenCover 𝒜).openCover.J
      e₁ : CategoryTheory.Iso (CategoryTheory.Limits.pullback (CategoryTheory.Catego …
      e₂ : CategoryTheory.Iso (CategoryTheory.Limits.pullback ((AlgebraicGeometry.Pr …
      F : RingHom (TensorProduct (Subtype fun x => Membership.mem (𝒜 0) x) (Homogene …
      this : Function.Surjective ⇑F
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Proj.pullbackAwayι …
    -/
    erw [pullbackAwayιIso_inv_snd]
    /-
      case h.e'_3.h₁
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      i j : (AlgebraicGeometry.Proj.affineOpenCover 𝒜).openCover.J
      e₁ : CategoryTheory.Iso (CategoryTheory.Limits.pullback (CategoryTheory.Catego …
      e₂ : CategoryTheory.Iso (CategoryTheory.Limits.pullback ((AlgebraicGeometry.Pr …
      F : RingHom (TensorProduct (Subtype fun x => Membership.mem (𝒜 0) x) (Homogene …
      this : Function.Surjective ⇑F
      ⊢ Eq (AlgebraicGeometry.Spec.map (CommRingCat.ofHom (HomogeneousLocalization.a …
    -/
    congr 1
    /-
      case h.e'_3.h₁.e_f
      R : Type u_1
      A : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing A
      inst✝¹ : Algebra R A
      𝒜 : Nat → Submodule R A
      inst✝ : GradedAlgebra 𝒜
      i j : (AlgebraicGeometry.Proj.affineOpenCover 𝒜).openCover.J
      e₁ : CategoryTheory.Iso (CategoryTheory.Limits.pullback (CategoryTheory.Catego …
      e₂ : CategoryTheory.Iso (CategoryTheory.Limits.pullback ((AlgebraicGeometry.Pr …
      F : RingHom (TensorProduct (Subtype fun x => Membership.mem (𝒜 0) x) (Homogene …
      this : Function.Surjective ⇑F
      ⊢ Eq (CommRingCat.ofHom (HomogeneousLocalization.awayMap 𝒜 ⋯ ⋯)) (CategoryTheo …
    -/
    ext x : 2
    exact DFunLike.congr_fun (Algebra.TensorProduct.lift_comp_includeRight
      (awayMapₐ 𝒜 j.2.2 rfl) (awayMapₐ 𝒜 i.2.2 (mul_comm _ _)) (fun _ _ ↦ .all _ _)).symm x


@[stacks 01MC]
instance : Scheme.IsSeparated (Proj 𝒜) :=
  (HasAffineProperty.iff_of_isAffine (P := @IsSeparated)).mp (isSeparated 𝒜)


