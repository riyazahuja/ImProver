/-- We equip the sphere, in a seminormed group, with a formal operation of negation, namely the
antipodal map. -/
instance : InvolutiveNeg (sphere (0 : E) r) where
                                         /-
                                           E : Type u_1
                                           i : SeminormedAddCommGroup E
                                           r : Real
                                           w : E
                                           ⊢ Membership.mem (Metric.sphere 0 r) w → Membership.mem (Metric.sphere 0 r) (N …
                                         -/
  neg := Subtype.map Neg.neg fun w => by simp
                                         /-
                                           🎉 no goals
                                         -/
  neg_neg x := Subtype.ext <| neg_neg x.1


@[simp]
theorem coe_neg_sphere {r : ℝ} (v : sphere (0 : E) r) : ↑(-v) = (-v : E) :=
  rfl


instance : ContinuousNeg (sphere (0 : E) r) := IsInducing.subtypeVal.continuousNeg fun _ => rfl


/-- We equip the ball, in a seminormed group, with a formal operation of negation, namely the
antipodal map. -/
instance {r : ℝ} : InvolutiveNeg (ball (0 : E) r) where
                                         /-
                                           E : Type u_1
                                           i : SeminormedAddCommGroup E
                                           r✝ r : Real
                                           w : E
                                           ⊢ Membership.mem (Metric.ball 0 r) w → Membership.mem (Metric.ball 0 r) (Neg.n …
                                         -/
  neg := Subtype.map Neg.neg fun w => by simp
                                         /-
                                           🎉 no goals
                                         -/
  neg_neg x := Subtype.ext <| neg_neg x.1


@[simp] theorem coe_neg_ball {r : ℝ} (v : ball (0 : E) r) : ↑(-v) = (-v : E) := rfl


instance : ContinuousNeg (ball (0 : E) r) := IsInducing.subtypeVal.continuousNeg fun _ => rfl


/-- We equip the closed ball, in a seminormed group, with a formal operation of negation, namely the
antipodal map. -/
instance {r : ℝ} : InvolutiveNeg (closedBall (0 : E) r) where
                                         /-
                                           E : Type u_1
                                           i : SeminormedAddCommGroup E
                                           r✝ r : Real
                                           w : E
                                           ⊢ Membership.mem (Metric.closedBall 0 r) w → Membership.mem (Metric.closedBall …
                                         -/
  neg := Subtype.map Neg.neg fun w => by simp
                                         /-
                                           🎉 no goals
                                         -/
  neg_neg x := Subtype.ext <| neg_neg x.1


@[simp] theorem coe_neg_closedBall {r : ℝ} (v : closedBall (0 : E) r) : ↑(-v) = (-v : E) := rfl


instance : ContinuousNeg (closedBall (0 : E) r) := IsInducing.subtypeVal.continuousNeg  fun _ => rfl

